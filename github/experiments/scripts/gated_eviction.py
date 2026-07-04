"""Gated KV-cache eviction.

Algorithm:
  1. Prefill the prompt (free).
  2. Compute the head-agreement drop (early-vs-late thirds, mean Jaccard top-K).
  3. If drop >= threshold tau: apply SnapKV-style eviction at budget b.
     Else (capacity-bound input): keep full KV.

This is a one-line wrapper around any base eviction method that prevents
catastrophic failure on retrieval-precision tasks. The threshold tau is
calibrated once per (model, eviction-baseline) on a held-out validation
split.

Usage examples:
  # Run gated-SnapKV at fixed budget 0.5 with threshold 0.05
  python gated_eviction.py --config 4096 --tasks niah_multikey_3,vt,fwe,qa_1 \
    --budgets 1.0,0.5,0.25 --tau 0.05 --out results/gated.jsonl
"""
import argparse
import itertools
import json
import os
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


def build_prompt(tokenizer, ex):
    user_msg = ex["context"] + "\n\n" + ex["question"]
    if ex.get("answer_prefix"):
        user_msg = user_msg + "\n" + ex["answer_prefix"]
    messages = [
        {"role": "system", "content": "Answer the question concisely."},
        {"role": "user", "content": user_msg},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def jaccard(s1, s2):
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return inter / union if union else 0.0


def head_agreement_layer(attn_layer, top_k):
    """attn_layer: [H, W, T]. Returns mean Jaccard top-k across head pairs."""
    H, W, T = attn_layer.shape
    head_mean = attn_layer.mean(dim=1)  # [H, T]
    top_idx = head_mean.topk(min(top_k, T), dim=-1).indices
    sets = [set(top_idx[h].cpu().tolist()) for h in range(H)]
    pairs = list(itertools.combinations(range(H), 2))
    if not pairs:
        return 0.0
    return sum(jaccard(sets[i], sets[j]) for i, j in pairs) / len(pairs)


def compute_drop_from_attentions(attentions, obs_window, top_k):
    """attentions: tuple of L layers, each [B, H, T, T] or (after two-pass) [B, H, obs_window, T]."""
    L = len(attentions)
    per_layer = []
    for a in attentions:
        if a.shape[2] != obs_window:
            window = a[0, :, -obs_window:, :]  # one-pass case
        else:
            window = a[0]  # two-pass case (already obs_window queries)
        per_layer.append(head_agreement_layer(window, top_k))
    third = max(1, L // 3)
    early = sum(per_layer[:third]) / third
    late = sum(per_layer[L - third:]) / third
    return early - late, per_layer


def derive_keep_mask(score, T, budget_ratio, obs_window, n_sink, device,
                     policy="snapkv"):
    """Build a [T] bool keep mask given a per-position score.

    For policy=='streamingllm' we override the middle selection entirely:
    only sink + recent tokens are kept (no middle), regardless of the score
    or budget. This is the standard StreamingLLM keep set; any "extra"
    budget is simply unused (effective n_kept = n_sink + obs_window).
    """
    if budget_ratio >= 1.0:
        return torch.ones(T, dtype=torch.bool, device=device)

    keep = torch.zeros(T, dtype=torch.bool, device=device)
    keep[:n_sink] = True
    keep[T - obs_window:] = True

    if policy == "streamingllm":
        # No middle tokens — StreamingLLM is sink + recent only.
        return keep

    budget = max(obs_window + n_sink + 4, int(T * budget_ratio))
    n_middle = max(0, budget - n_sink - obs_window)
    cand_start, cand_end = n_sink, T - obs_window
    if n_middle > 0 and cand_end > cand_start:
        cs = score[cand_start:cand_end]
        k = min(n_middle, cs.shape[0])
        topk = torch.topk(cs, k).indices
        keep[cand_start + topk] = True
    return keep


def derive_keep_masks_per_layer(scores, T, budget_ratio, obs_window, n_sink, device):
    """scores: [L, T]. Returns [L, T] bool mask; every row keeps the same
    n_kept (budget-derived), only the middle positions differ per layer."""
    L = scores.shape[0]
    masks = torch.zeros(L, T, dtype=torch.bool, device=device)
    for i in range(L):
        masks[i] = derive_keep_mask(scores[i], T, budget_ratio, obs_window,
                                    n_sink, device, policy="topk")
    return masks


def derive_keep_masks_per_head(dist, T, budget_ratio, obs_window, n_sink, device):
    """dist: [H, T]. Returns [H, T] bool mask; every head keeps the same
    n_kept (budget-derived), only the middle positions differ per head."""
    H = dist.shape[0]
    masks = torch.zeros(H, T, dtype=torch.bool, device=device)
    for h in range(H):
        masks[h] = derive_keep_mask(dist[h], T, budget_ratio, obs_window,
                                    n_sink, device, policy="topk")
    return masks


def manifoldkv_score(full_keys, device):
    """ManifoldKV-style geometric eviction score (arXiv 2602.08343 variant).

    Per (layer, kv-head): centroid = mean key over positions; the score of a
    position is the Euclidean (L2) distance of its key vector to that
    centroid. Higher distance = geometric outlier = KEEP (retains outlier
    keys, avoiding the "directional collisions" of attention-score methods
    on multi-key retrieval). Distances are computed in fp32.

    Deviations from the paper, forced by this codebase's single-mask-across-
    layers design (uniform cache shape so the standard DynamicCache
    generation path works — same constraint documented for pyramidkv):
      - The paper scores (and can evict) per layer; we average the per-
        (layer, head) distances into ONE [T] score and derive a single keep
        mask shared by all layers.
      - Raw (unnormalized) distances are averaged, so heads/layers with
        larger key norms contribute more to the pooled score. The paper's
        abstract specifies unnormalized Euclidean distance ("both angular
        and radial deviations"), so we do NOT normalize keys; only the
        cross-head/layer pooling is our addition.
    Sink/recency protection is identical to the other policies (handled in
    derive_keep_mask), and the head-agreement GATE signal is still computed
    from attentions, independent of this scorer.
    """
    per_layer = []
    for k in full_keys:  # [1, H_kv, T, head_dim]
        kf = k[0].float()                         # [H, T, D]
        centroid = kf.mean(dim=1, keepdim=True)   # [H, 1, D]
        dist = (kf - centroid).norm(dim=-1)       # [H, T]
        per_layer.append(dist.mean(dim=0))        # [T]
    return torch.stack(per_layer).mean(dim=0).to(device)


def manifoldkv_perlayer_score(full_keys, device):
    """TRUE per-layer ManifoldKV score: returns a [L, T] tensor.

    For each layer, the position score is the mean over kv-heads of the
    Euclidean (L2) distance of that position's key vector to the head's key
    centroid (fp32). No cross-layer pooling: each layer keeps its OWN outlier
    positions (all heads within a layer share the layer's keep-set, chosen by
    averaging the per-head distances). This is faithful to ManifoldKV's
    per-layer independence; the only remaining deviation from the paper is
    that heads within a layer share the layer keep-set (the per-head variant
    is manifoldkv_perhead). Uniform n_kept per layer (budget * T) keeps the
    DynamicCache rectangular so the standard decode path works.
    """
    per_layer = []
    for k in full_keys:  # [1, H_kv, T, head_dim]
        kf = k[0].float()                         # [H, T, D]
        centroid = kf.mean(dim=1, keepdim=True)   # [H, 1, D]
        dist = (kf - centroid).norm(dim=-1)       # [H, T]
        per_layer.append(dist.mean(dim=0))        # [T]
    return torch.stack(per_layer).to(device)      # [L, T]


def manifoldkv_perhead_dists(full_keys, device):
    """Most-faithful per-(layer, kv-head) ManifoldKV distances.

    Returns a list (length L) of [H_kv, T] fp32 distance tensors: for each
    (layer, kv-head), the L2 distance of every position's key to that head's
    own centroid. Each kv-head then keeps its OWN outlier positions (an
    independent keep-set per (layer, head)). Because the budget, sink and
    recency window are shared, every head keeps the SAME number of positions
    (only the positions differ), so per-head gathering yields a rectangular
    [1, H_kv, n_kept, D] cache tensor per layer -> the standard decode path
    still works (no ragged/padded surgery). Query heads in a GQA group share
    their kv-head's keep-set, which is consistent.
    """
    out = []
    for k in full_keys:  # [1, H_kv, T, head_dim]
        kf = k[0].float()                         # [H, T, D]
        centroid = kf.mean(dim=1, keepdim=True)   # [H, 1, D]
        dist = (kf - centroid).norm(dim=-1)       # [H, T]
        out.append(dist.to(device))
    return out


def pool_score(attentions, policy, obs_window, T, device):
    """Pool per-layer attentions into a [T] per-position importance score.

    Policies:
      - snapkv: mean over last obs_window queries (the standard SnapKV score),
        averaged equally across layers.
      - h2o: mean over ALL queries (cumulative attention mass — the
        "heavy hitter" score). In the two-pass setting we only have the last
        obs_window queries' attention, so h2o degrades to snapkv there; we
        warn in the docstring.
      - streamingllm: returns zeros (score is unused — keep_mask is
        position-based; see derive_keep_mask).
      - pyramidkv: SnapKV-style per-layer score but pooled with DEPTH-DECREASING
        weights — bottom layer weight 2.0, top layer weight 0.5, linear
        interpolation. The pooled score then biases the (single, uniform)
        keep_mask toward tokens favored by the LOWER layers. This matches
        PyramidKV's intent ("bottom layers keep more of their high-attention
        tokens") while preserving uniform cache shape across layers so that
        downstream generation works with the standard DynamicCache path. The
        per-layer-budget variant of PyramidKV would require mixed-length
        caches across layers, which the transformers generation path does not
        support without invasive patching.
      - random: uniform random in [0,1).
    """
    if policy == "streamingllm":
        return torch.zeros(T, device=device)

    if policy == "random":
        g = torch.Generator(device=device)
        return torch.rand(T, generator=g, device=device)

    # Build per-layer [L, T] base scores from attentions.
    layer_scores = []
    for a in attentions:
        if policy == "h2o" and a.shape[2] > obs_window:
            # All queries to each key.
            layer_scores.append(a[0].mean(dim=(0, 1)))
        else:
            # SnapKV-style: last obs_window queries to each key.
            if a.shape[2] != obs_window:
                w = a[0, :, -obs_window:, :]
            else:
                w = a[0]
            layer_scores.append(w.mean(dim=(0, 1)))
    layer_scores = torch.stack(layer_scores)  # [L, T]
    L = layer_scores.shape[0]

    if policy == "pyramidkv":
        # Bottom layer weight 2.0, top layer weight 0.5; linear interp.
        weights = torch.linspace(2.0, 0.5, L, device=device).unsqueeze(1)
        return (layer_scores * weights).sum(dim=0) / weights.sum()

    # snapkv (default) and h2o both reduce by equal mean over layers.
    return layer_scores.mean(dim=0)


def compute_score(score_policy, full_keys, attentions, obs_window, T, device):
    """Dispatch to the right scorer. Returns:
      - manifoldkv           -> [T] tensor (single shared mask)
      - manifoldkv_perlayer  -> [L, T] tensor (independent per-layer keep-sets)
      - manifoldkv_perhead   -> list of L tensors [H, T] (per-(layer,head) keep-sets)
      - everything else       -> [T] attention-pooled tensor
    """
    if score_policy == "manifoldkv":
        return manifoldkv_score(full_keys, device)
    if score_policy == "manifoldkv_perlayer":
        return manifoldkv_perlayer_score(full_keys, device)
    if score_policy == "manifoldkv_perhead":
        return manifoldkv_perhead_dists(full_keys, device)
    return pool_score(attentions, score_policy, obs_window, T, device)


def prefill_and_score_with_agreement(model, ids, obs_window, top_k, device, two_pass,
                                     score_policy="snapkv"):
    """Prefill, return (full_keys, full_values, score [T], drop, per_layer_agreement).

    Two-pass version (for long contexts): flash prefill without attentions, then
    a tiny re-forward of the last obs_window queries with output_attentions=True
    to get the scoring + agreement signal cheaply. In two-pass mode the
    attention matrix only has obs_window queries, so the h2o "all queries"
    score degrades to snapkv there (this is documented in pool_score).
    """
    T = ids.shape[1]

    # Detect whether the model was loaded with sdpa (long-context safe path)
    loaded_impl = getattr(model.config, "_attn_implementation", "eager")

    def switch_layers(impl):
        for layer in model.model.layers:
            for attr_name in ("self_attn", "attention"):
                attn_mod = getattr(layer, attr_name, None)
                if attn_mod is not None:
                    try:
                        attn_mod.config._attn_implementation = impl
                    except AttributeError:
                        pass

    if not two_pass:
        with torch.no_grad():
            out = model(
                input_ids=ids,
                output_attentions=True,
                use_cache=True,
                return_dict=True,
            )
        past = out.past_key_values
        full_keys = [layer.keys.clone() for layer in past.layers]
        full_values = [layer.values.clone() for layer in past.layers]
        score = compute_score(score_policy, full_keys, out.attentions, obs_window, T, device)
        drop, per_layer = compute_drop_from_attentions(out.attentions, obs_window, top_k)
        del out, past
        torch.cuda.empty_cache()
        return full_keys, full_values, score, drop, per_layer

    # two_pass path
    with torch.no_grad():
        prefill = model(
            input_ids=ids,
            output_attentions=False,
            use_cache=True,
            return_dict=True,
        )
    past = prefill.past_key_values
    full_keys = [layer.keys.clone() for layer in past.layers]
    full_values = [layer.values.clone() for layer in past.layers]
    del prefill, past
    torch.cuda.empty_cache()

    past_short = DynamicCache()
    for i, (k, v) in enumerate(zip(full_keys, full_values)):
        past_short.update(
            k[:, :, : T - obs_window, :].clone(),
            v[:, :, : T - obs_window, :].clone(),
            i,
        )
    last_ids = ids[:, -obs_window:]
    pos_ids = torch.arange(T - obs_window, T, device=device).unsqueeze(0)
    cache_pos = torch.arange(T - obs_window, T, device=device)

    # Pass-2 needs output_attentions; force eager just for this small call.
    switch_layers("eager")
    try:
        with torch.no_grad():
            scoring = model(
                input_ids=last_ids,
                past_key_values=past_short,
                position_ids=pos_ids,
                cache_position=cache_pos,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
    finally:
        switch_layers(loaded_impl)
    score = compute_score(score_policy, full_keys, scoring.attentions, obs_window, T, device)
    drop, per_layer = compute_drop_from_attentions(scoring.attentions, obs_window, top_k)
    del scoring, past_short
    torch.cuda.empty_cache()
    return full_keys, full_values, score, drop, per_layer


def generate_from_past(model, tokenizer, past, ids, max_new, device, original_T):
    last_prompt_id = ids[:, -1:].clone()
    n_kept = past.layers[0].keys.shape[2]

    out_ids = []
    next_token = last_prompt_id
    next_position = torch.tensor([[original_T - 1]], dtype=torch.long, device=device)
    cache_position = torch.tensor([n_kept], dtype=torch.long, device=device)

    for step in range(max_new):
        with torch.no_grad():
            out = model(
                input_ids=next_token,
                past_key_values=past,
                position_ids=next_position,
                cache_position=cache_position,
                use_cache=True,
                return_dict=True,
            )
        past = out.past_key_values
        new_id = int(out.logits[:, -1, :].argmax(dim=-1).item())
        out_ids.append(new_id)
        if new_id == tokenizer.eos_token_id:
            break
        next_token = torch.tensor([[new_id]], dtype=torch.long, device=device)
        next_position = next_position + 1
        cache_position = cache_position + 1

    return tokenizer.decode(out_ids, skip_special_tokens=True).strip()


def is_correct(pred, golds, task=None):
    pred_lower = pred.lower()
    if task in ("vt", "fwe", "cwe", "niah_multivalue"):
        return all(g.strip().lower() in pred_lower for g in golds)
    return any(g.strip().lower() in pred_lower for g in golds)


def evaluate_one(model, tokenizer, ids, full_keys, full_values, score, T,
                 budget_ratio, max_new, device, obs_window, n_sink, do_evict,
                 score_policy="snapkv"):
    """Evaluate one budget. If do_evict is False, use full KV regardless of budget.

    Supports three cache-slicing modes:
      - uniform single mask (snapkv/h2o/pyramidkv/manifoldkv/...): one [T] mask
        shared by all layers/heads.
      - manifoldkv_perlayer: an independent [T] keep-set per layer (same count,
        different positions), sliced per layer.
      - manifoldkv_perhead: an independent keep-set per (layer, kv-head), gathered
        per head into a rectangular [1, H, n_kept, D] tensor per layer.
    """
    evict = do_evict and budget_ratio < 1.0
    past_b = DynamicCache()

    if not evict:
        keep_idx = torch.arange(T, device=device)
        for i, (k, v) in enumerate(zip(full_keys, full_values)):
            past_b.update(k.index_select(2, keep_idx), v.index_select(2, keep_idx), i)
        n_kept = T

    elif score_policy == "manifoldkv_perlayer":
        masks = derive_keep_masks_per_layer(score, T, budget_ratio, obs_window,
                                            n_sink, device)  # [L, T]
        n_kept = None
        for i, (k, v) in enumerate(zip(full_keys, full_values)):
            idx_i = masks[i].nonzero(as_tuple=True)[0]
            if n_kept is None:
                n_kept = int(idx_i.shape[0])
            else:
                assert int(idx_i.shape[0]) == n_kept, "per-layer n_kept differs"
            past_b.update(k.index_select(2, idx_i), v.index_select(2, idx_i), i)

    elif score_policy == "manifoldkv_perhead":
        n_kept = None
        for i, (k, v) in enumerate(zip(full_keys, full_values)):
            masks_h = derive_keep_masks_per_head(score[i], T, budget_ratio,
                                                 obs_window, n_sink, device)  # [H, T]
            H = masks_h.shape[0]
            idx_list = [masks_h[h].nonzero(as_tuple=True)[0] for h in range(H)]
            nkh = int(idx_list[0].shape[0])
            for il in idx_list:
                assert int(il.shape[0]) == nkh, "per-head n_kept differs"
            if n_kept is None:
                n_kept = nkh
            idx = torch.stack(idx_list, dim=0)                     # [H, nkh]
            D = k.shape[-1]
            gidx = idx.unsqueeze(0).unsqueeze(-1).expand(1, H, nkh, D)
            k_g = torch.gather(k, 2, gidx)
            v_g = torch.gather(v, 2, gidx)
            past_b.update(k_g, v_g, i)

    else:
        keep_mask = derive_keep_mask(score, T, budget_ratio, obs_window, n_sink,
                                     device, policy=score_policy)
        keep_idx = keep_mask.nonzero(as_tuple=True)[0]
        n_kept = int(keep_idx.shape[0])
        for i, (k, v) in enumerate(zip(full_keys, full_values)):
            past_b.update(k.index_select(2, keep_idx), v.index_select(2, keep_idx), i)

    pred = generate_from_past(model, tokenizer, past_b, ids, max_new, device, original_T=T)
    del past_b
    return pred, n_kept


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="4096")
    p.add_argument("--tasks", default="niah_multikey_3,vt,fwe,qa_1")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--budgets", default="1.0,0.5,0.25,0.125")
    p.add_argument("--max_examples", type=int, default=100)
    p.add_argument("--max_new", type=int, default=128)
    p.add_argument("--obs_window", type=int, default=32)
    p.add_argument("--n_sink", type=int, default=4)
    p.add_argument("--top_k", type=int, default=32, help="top-k for head agreement")
    p.add_argument("--tau", type=float, default=0.05,
                   help="threshold on agreement drop; if drop < tau, skip eviction")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--two_pass", action="store_true")
    p.add_argument("--attn_impl", default="eager", choices=["eager", "sdpa"],
                   help="Loaded attn implementation. For long contexts (T>=16K), use sdpa to avoid OOM.")
    p.add_argument("--score_policy", default="snapkv",
                   choices=["snapkv", "h2o", "streamingllm", "pyramidkv", "manifoldkv",
                            "manifoldkv_perlayer", "manifoldkv_perhead", "random"],
                   help="Eviction scoring policy. snapkv = mean attention from last obs_window; "
                        "h2o = mean attention from all queries (cumulative heavy-hitter mass); "
                        "streamingllm = sink+recent only, no middle tokens; "
                        "pyramidkv = SnapKV score pooled with bottom-layer-heavy weights "
                        "(uniform cache shape; see pool_score docstring); "
                        "manifoldkv = attention-free geometric score: Euclidean distance of "
                        "each key to the per-(layer, kv-head) key centroid, averaged over "
                        "(layer, head) into ONE shared mask (see manifoldkv_score docstring); "
                        "manifoldkv_perlayer = TRUE per-layer ManifoldKV: independent keep-set "
                        "per layer (heads in a layer share it, distances averaged over heads); "
                        "manifoldkv_perhead = most-faithful per-(layer, kv-head) keep-set "
                        "(each kv-head keeps its own outliers, uniform count); "
                        "random = uniform random.")
    args = p.parse_args()

    budgets = [float(b) for b in args.budgets.split(",")]
    tasks = args.tasks.split(",")
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"loading RULER {args.config}")
    ds = load_dataset("simonjegou/ruler", args.config, split="test")
    rows = []
    for task in tasks:
        sub = ds.filter(lambda r: r["task"] == task).select(range(min(args.max_examples, len(ds))))
        for r in sub:
            rows.append(r)
    print(f"loaded {len(rows)} rows across tasks {tasks}")
    print(f"using tau = {args.tau}")

    print(f"loading {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
        trust_remote_code=True,
    ).to(device).eval()

    results_f = open(args.out, "w")
    t0 = time.time()
    n_gate_open_running = 0
    for ex_i, ex in enumerate(rows):
        prompt = build_prompt(tokenizer, ex)
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        T = ids.shape[1]

        full_keys, full_values, score, drop, per_layer = prefill_and_score_with_agreement(
            model, ids, args.obs_window, args.top_k, device, two_pass=args.two_pass,
            score_policy=args.score_policy,
        )
        gate_open = drop >= args.tau  # True => apply eviction; False => keep full KV
        if gate_open:
            n_gate_open_running += 1

        max_new = int(ex.get("max_new_tokens") or args.max_new)
        max_new = max(max_new, args.max_new)

        for b in budgets:
            # Two policies for direct comparison:
            #  - plain: always apply eviction at b (or full KV if b==1.0)
            #  - gated: apply eviction at b iff gate_open
            pred_plain, n_kept_plain = evaluate_one(
                model, tokenizer, ids, full_keys, full_values, score, T,
                b, max_new, device, args.obs_window, args.n_sink, do_evict=True,
                score_policy=args.score_policy,
            )
            pred_gated, n_kept_gated = evaluate_one(
                model, tokenizer, ids, full_keys, full_values, score, T,
                b, max_new, device, args.obs_window, args.n_sink, do_evict=bool(gate_open),
                score_policy=args.score_policy,
            )

            ok_plain = is_correct(pred_plain, ex["answer"], task=ex["task"])
            ok_gated = is_correct(pred_gated, ex["answer"], task=ex["task"])

            rec = {
                "id": ex_i,
                "task": ex["task"],
                "budget": b,
                "T": T,
                "drop": float(drop),
                "gate_open": bool(gate_open),
                "score_policy": args.score_policy,
                "n_kept_plain": n_kept_plain,
                "n_kept_gated": n_kept_gated,
                "gold": ex["answer"],
                "pred_plain": pred_plain[:200],
                "pred_gated": pred_gated[:200],
                "correct_plain": bool(ok_plain),
                "correct_gated": bool(ok_gated),
            }
            results_f.write(json.dumps(rec) + "\n")
            results_f.flush()

        del full_keys, full_values, score
        torch.cuda.empty_cache()

        if (ex_i + 1) % 5 == 0:
            elapsed = time.time() - t0
            rate = (ex_i + 1) / elapsed
            eta = (len(rows) - ex_i - 1) / rate
            n_gate_open = n_gate_open_running / max(1, ex_i + 1)
            print(f"  [{ex_i+1}/{len(rows)}]  task={ex['task']}  T={T}  drop={drop:+.4f}  "
                  f"gate_open={gate_open}  elapsed={elapsed:.0f}s  eta={eta:.0f}s  "
                  f"gate_open_frac={n_gate_open:.3f}")

    results_f.close()
    print(f"done in {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
