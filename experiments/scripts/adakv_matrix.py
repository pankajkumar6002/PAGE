"""E8 (W2, strongest form): the headline matrix with a per-head Ada-KV plain arm.

The paper's plain arm derives ONE keep-mask per layer and shares it across
heads. Its own tab:matrix caption concedes this overstates the collapse at
moderate budgets ("a clean per-head SnapKV collapses less sharply at 2x, 0.32
not 0.14 on MK3"), and Limitations calls the result "partly a single-mask
artifact". This script replaces the plain arm with per-head Ada-KV allocation
and recomputes the gated-minus-plain delta on the same inputs.

Design notes, all load-bearing:

* Four arms are recorded per (input, budget) in ONE run on the SAME inputs, so
  every comparison is paired: full cache, plain-single, plain-adakv, and the
  two gated arms derived from them. No cross-run alignment is needed.

* The gate signal D is computed from the prefill attentions, which do not
  depend on the allocation policy, so D must reproduce the released logs
  exactly. That is asserted downstream (pre-registration P5), not assumed.

* Allocation is the ONLY thing that varies. Both plain arms use identical
  SnapKV scores, identical sink/recency protection, and the same total token
  budget (Ada-KV redistributes across heads, it does not spend more).

* Ada-KV yields ragged per-head keep counts, which HF DynamicCache cannot store
  rectangularly. Following the faithful reproduction in the original repo, we
  keep the full cache and mask attention per kv-head, which reproduces exactly
  the logits a ragged compressed cache would produce. This measures accuracy
  under a logical budget, which is what W2 asks about, not wall-clock memory.

Usage (see gpu/run_adakv.sh):
  python adakv_matrix.py --model Qwen/Qwen2.5-1.5B-Instruct --slug qwen15b \
    --config 4096 --tasks niah_multikey_3,vt,fwe,qa_1 --max_examples 100 \
    --budgets 0.0625,0.125,0.25,0.375,0.5,0.625,0.75,0.875 --tau 0.07 \
    --out .../experiments/results/adakv_4k_qwen15b.jsonl
"""
import argparse
import itertools
import json
import math
import os
import sys
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Per-kv-head attention masking, patched per model family.
# ---------------------------------------------------------------------------
_STATE = {"active": False, "head_mask": None}
_PATCHED = []


def _make_patched(mod):
    orig = mod.eager_attention_forward

    def patched(module, query, key, value, attention_mask, scaling,
                dropout=0.0, **kwargs):
        key_states = mod.repeat_kv(key, module.num_key_value_groups)
        value_states = mod.repeat_kv(value, module.num_key_value_groups)
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[..., : key_states.shape[-2]]
        if _STATE["active"]:
            keep = _STATE["head_mask"][module.layer_idx]  # [n_kv, T] bool
            # Under device_map the layers are spread across cards, so the mask
            # must follow the attention tensor rather than the card it was
            # built on.
            if keep.device != attn_weights.device:
                keep = keep.to(attn_weights.device)
            n_kv, T = keep.shape
            cur_kv = attn_weights.shape[-1]
            keep_q = keep.repeat_interleave(module.num_key_value_groups, dim=0)
            if cur_kv > T:  # decoded tokens are always kept (recency)
                pad = torch.ones(keep_q.shape[0], cur_kv - T,
                                 dtype=torch.bool, device=keep_q.device)
                keep_q = torch.cat([keep_q, pad], dim=1)
            elif cur_kv < T:
                keep_q = keep_q[:, :cur_kv]
            add = torch.zeros_like(keep_q, dtype=attn_weights.dtype)
            add.masked_fill_(~keep_q, float("-inf"))
            attn_weights = attn_weights + add[None, :, None, :]
        attn_weights = torch.nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = torch.nn.functional.dropout(
            attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states)
        return attn_output.transpose(1, 2).contiguous(), attn_weights

    return orig, patched


def patch_families():
    """Patch every family we run.

    Llama is required for the E8 bias control: Llama-3.1-8B has the same
    (L=32, Q=32, KV=8) shape as Mistral-7B but a different family, so it
    separates "8 KV-heads" from "Mistral-specific" as the cause of the P1
    violation. WITHOUT the patch the per-head mask silently does not apply and
    the two arms would be identical, which would look like a null result.
    """
    import transformers.models.qwen2.modeling_qwen2 as qwen2_mod
    import transformers.models.mistral.modeling_mistral as mistral_mod
    import transformers.models.llama.modeling_llama as llama_mod
    for mod in (qwen2_mod, mistral_mod, llama_mod):
        orig, patched = _make_patched(mod)
        mod.eager_attention_forward = patched
        if hasattr(mod, "ALL_ATTENTION_FUNCTIONS"):
            try:
                mod.ALL_ATTENTION_FUNCTIONS.register("eager", patched)
            except Exception:
                pass
        _PATCHED.append((mod, orig))


# ---------------------------------------------------------------------------
# Gate signal D. Copied verbatim from gated_eviction.py so the statistic is
# byte-identical; asserted against the released logs downstream.
# ---------------------------------------------------------------------------
def jaccard(s1, s2):
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return inter / union if union else 0.0


def head_agreement_layer(attn_layer, top_k):
    """attn_layer: [H, W, T]. Mean pairwise top-k Jaccard over head pairs."""
    H, W, T = attn_layer.shape
    head_mean = attn_layer.mean(dim=1)
    top_idx = head_mean.topk(min(top_k, T), dim=-1).indices
    sets = [set(top_idx[h].cpu().tolist()) for h in range(H)]
    pairs = list(itertools.combinations(range(H), 2))
    return sum(jaccard(sets[i], sets[j]) for i, j in pairs) / len(pairs)


def compute_drop(attentions, obs_window, top_k):
    per_layer = []
    for a in attentions:
        w = a[0, :, -obs_window:, :] if a.shape[2] != obs_window else a[0]
        per_layer.append(head_agreement_layer(w, top_k))
    L = len(per_layer)
    third = max(1, L // 3)
    early = sum(per_layer[:third]) / third
    late = sum(per_layer[L - third:]) / third
    return early - late, per_layer


# ---------------------------------------------------------------------------
# Prompt / scoring helpers, mirroring gated_eviction.py exactly.
# ---------------------------------------------------------------------------
def build_prompt(tokenizer, ex):
    """Canonical prompt. Reimplementing this drifted D by 4e-4 (a `\\n\\n` vs
    `\\n` before the answer prefix), so it mirrors gated_eviction.py exactly."""
    user_msg = ex["context"] + "\n\n" + ex["question"]
    if ex.get("answer_prefix"):
        user_msg = user_msg + "\n" + ex["answer_prefix"]
    messages = [
        {"role": "system", "content": "Answer the question concisely."},
        {"role": "user", "content": user_msg},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)


def is_correct(pred, golds, task=None):
    pred_lower = pred.lower()
    if task in ("vt", "fwe", "cwe", "niah_multivalue"):
        return all(g.strip().lower() in pred_lower for g in golds)
    return any(g.strip().lower() in pred_lower for g in golds)


# ---------------------------------------------------------------------------
# Allocation. `uniform` reproduces the shared-mask arm; `adakv` is the per-head
# redistribution. Total budget is identical between them.
# ---------------------------------------------------------------------------
def build_keep_mask(scores, budget_ratio, obs_window, n_sink, alloc,
                    floor_alpha, device):
    """scores: [n_kv, T] (higher = keep). Returns [n_kv, T] bool, mean kept."""
    n_kv, T = scores.shape
    keep = torch.zeros(n_kv, T, dtype=torch.bool, device=device)
    if budget_ratio >= 1.0:
        keep[:] = True
        return keep, float(T)
    keep[:, :n_sink] = True
    keep[:, T - obs_window:] = True

    target_per_head = max(n_sink + obs_window, int(round(budget_ratio * T)))
    mid_per_head = target_per_head - (n_sink + obs_window)
    cand_start, cand_end = n_sink, T - obs_window
    n_cand = cand_end - cand_start
    if mid_per_head <= 0 or n_cand <= 0:
        return keep, float(keep.sum(dim=1).float().mean())
    mid_per_head = min(mid_per_head, n_cand)
    sc = scores[:, cand_start:cand_end]

    if alloc == "shared":
        # One mask for every head: pool the scores across heads first. This is
        # the paper's plain arm.
        pooled = sc.mean(dim=0)                       # [n_cand]
        idx = torch.topk(pooled, mid_per_head).indices
        keep[:, cand_start + idx] = True
    elif alloc == "uniform":
        idx = torch.topk(sc, mid_per_head, dim=1).indices
        for h in range(n_kv):
            keep[h, cand_start + idx[h]] = True
    elif alloc == "adakv":
        M_total = mid_per_head * n_kv
        floor_mid = min(int(math.floor(floor_alpha * mid_per_head)), n_cand)
        work = sc.clone()
        if floor_mid > 0:
            fidx = torch.topk(work, floor_mid, dim=1).indices
            for h in range(n_kv):
                keep[h, cand_start + fidx[h]] = True
                work[h, fidx[h]] = float("-inf")
        remaining = M_total - floor_mid * n_kv
        if remaining > 0:
            flat = work.reshape(-1)
            remaining = min(remaining, int(torch.isfinite(flat).sum().item()))
            if remaining > 0:
                gidx = torch.topk(flat, remaining).indices
                keep[gidx // n_cand, cand_start + (gidx % n_cand)] = True
    else:
        raise ValueError(alloc)
    return keep, float(keep.sum(dim=1).float().mean())


# ---------------------------------------------------------------------------
# Prefill. One pass gives both the attentions (for D) and the KV cache, but the
# full attention tensor is L*H*T*T*2 bytes, which is 64 GB for Qwen2.5-14B at
# 4K and does not fit alongside 28 GB of weights on an 80 GB card. The two-pass
# path prefills without attentions, then re-forwards only the last obs_window
# queries against the cache. This is the same workaround the released 14B cell
# used (gated_eviction.py --two_pass), so the arms stay comparable.
# ---------------------------------------------------------------------------
def _global_snapkv_score(attentions, obs_window):
    """The canonical plain-arm score: ONE [T] vector, mean over the last
    obs_window queries, over heads, then averaged EQUALLY ACROSS LAYERS.

    This is what gated_eviction.pool_score returns for snapkv, and it yields a
    single keep-mask shared by every layer and head. Deriving a per-LAYER mask
    instead (which an earlier version of this file did) gives the plain arm a
    per-layer adaptivity the paper's baseline does not have, making the shared
    arm artificially strong and inverting the E8 comparison.
    """
    layer_scores = []
    for a in attentions:
        x = a[0]                                      # [H, W, T]
        w = x[:, -obs_window:, :] if x.shape[1] != obs_window else x
        layer_scores.append(w.mean(dim=(0, 1)))       # [T]
    return torch.stack(layer_scores).mean(dim=0).float()   # [T]


def _scores_from_attn(attentions, post_keys, obs_window):
    """SnapKV scores: attention mass from the last obs_window queries, per kv-head."""
    scores = []
    for li in range(len(attentions)):
        a = attentions[li][0]                         # [H, W, T]
        w = a[:, -obs_window:, :] if a.shape[1] != obs_window else a
        per_head = w.mean(dim=1)                      # [H, T]
        n_q = per_head.shape[0]
        n_kv = post_keys[li].shape[1]
        if n_q != n_kv:                               # GQA: pool query heads
            per_head = per_head.reshape(n_kv, n_q // n_kv, -1).mean(dim=1)
        scores.append(per_head.float())
    return scores


def prefill(model, ids, obs_window, top_k, device, two_pass=False):
    _STATE["active"] = False
    T = ids.shape[1]
    if not two_pass:
        with torch.no_grad():
            out = model(ids, use_cache=True, output_attentions=True)
        drop, per_layer = compute_drop(out.attentions, obs_window, top_k)
        post_keys, values = _cache_tensors(out.past_key_values)
        scores = _scores_from_attn(out.attentions, post_keys, obs_window)
        gscore = _global_snapkv_score(out.attentions, obs_window)
        del out
        torch.cuda.empty_cache()
        return post_keys, values, scores, gscore, drop, per_layer, T

    # Pass 1: cache only, no attention tensors materialized.
    with torch.no_grad():
        out = model(ids, use_cache=True, output_attentions=False)
    post_keys, values = _cache_tensors(out.past_key_values)
    del out
    torch.cuda.empty_cache()

    # Pass 2: re-forward the last obs_window queries against the prefix cache,
    # which yields a [H, obs_window, T] attention block per layer.
    from transformers.cache_utils import DynamicCache
    prefix = DynamicCache()
    for li, (k, v) in enumerate(zip(post_keys, values)):
        prefix.update(k[:, :, : T - obs_window].clone(),
                      v[:, :, : T - obs_window].clone(), li)
    tail = ids[:, T - obs_window:]
    with torch.no_grad():
        out = model(tail, past_key_values=prefix, use_cache=True,
                    output_attentions=True)
    drop, per_layer = compute_drop(out.attentions, obs_window, top_k)
    scores = _scores_from_attn(out.attentions, post_keys, obs_window)
    gscore = _global_snapkv_score(out.attentions, obs_window)
    del out, prefix
    torch.cuda.empty_cache()
    return post_keys, values, scores, gscore, drop, per_layer, T


def _cache_tensors(cache):
    keys, values = [], []
    for layer in cache.layers:
        keys.append(layer.keys.detach())
        values.append(layer.values.detach())
    return keys, values


def generate(model, tokenizer, ids, post_keys, values, head_masks, max_new,
             device):
    """Decode greedily against the prefilled cache, per-head mask active.

    Mirrors gated_eviction.generate_from_past exactly. The loop is seeded with
    the LAST PROMPT TOKEN against an already-populated cache, and carries
    explicit position_ids / cache_position. Re-forwarding the whole prompt on
    step 0 instead double-counts it and desynchronises the positions, which
    silently truncated generations and pushed full-cache MK3 accuracy from
    0.65 down to 0.35.
    """
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    for li, (k, v) in enumerate(zip(post_keys, values)):
        cache.update(k.clone(), v.clone(), li)
    _STATE["active"] = head_masks is not None
    _STATE["head_mask"] = head_masks

    original_T = ids.shape[1]
    n_kept = cache.layers[0].keys.shape[2]
    out_ids = []
    next_token = ids[:, -1:].clone()
    next_position = torch.tensor([[original_T - 1]], dtype=torch.long,
                                 device=next_token.device)
    cache_position = torch.tensor([n_kept], dtype=torch.long,
                                  device=next_token.device)
    try:
        with torch.no_grad():
            for _ in range(max_new):
                out = model(input_ids=next_token, past_key_values=cache,
                            position_ids=next_position,
                            cache_position=cache_position,
                            use_cache=True, return_dict=True)
                cache = out.past_key_values
                new_id = int(out.logits[:, -1, :].argmax(dim=-1).item())
                out_ids.append(new_id)
                if new_id == tokenizer.eos_token_id:
                    break
                next_token = torch.tensor([[new_id]], dtype=torch.long,
                                          device=next_token.device)
                next_position = next_position + 1
                cache_position = cache_position + 1
    finally:
        _STATE["active"] = False
        _STATE["head_mask"] = None
    return tokenizer.decode(out_ids, skip_special_tokens=True).strip()


# Released full-cache accuracy per (slug, task), from
# page-kv/experiments/results/gated_4k_*.jsonl at b = 1.0. The runner asserts
# against these so a generation bug cannot masquerade as a method result.
RELEASED_FULL_ACC = {
    "qwen15b":   {"niah_multikey_3": 0.65, "vt": 0.82, "fwe": 0.22, "qa_1": 0.74},
    "qwen3b":    {"niah_multikey_3": 0.93, "vt": 1.00, "fwe": 0.76, "qa_1": 0.84},
    "qwen14b":   {"niah_multikey_3": 1.00, "vt": 1.00, "fwe": 0.92, "qa_1": 0.84},
    "mistral7b": {"niah_multikey_3": 0.99, "vt": 1.00, "fwe": 0.82, "qa_1": 0.81},
}
FULL_ACC_TOL = 0.08

# Released PLAIN-arm accuracy on NIAH-MK3 per (slug, budget), from the same
# logs. The full-cache guard above cannot catch a masking bug, because the
# full-cache arm uses no mask; nor can D, which comes from prefill attentions.
# These are the only released numbers that constrain the shared/plain arm, and
# a per-layer-vs-global mask defect inflated it ~10x before this check existed.
RELEASED_PLAIN_MK3 = {
    "qwen15b":   {0.0625: 0.00, 0.25: 0.02, 0.5: 0.14},
    "qwen3b":    {0.0625: 0.00, 0.25: 0.00, 0.5: 0.18},
    "qwen14b":   {0.0625: 0.00, 0.25: 0.00, 0.5: 0.60},
    "mistral7b": {0.0625: 0.00, 0.25: 0.00, 0.5: 0.12},
}
PLAIN_ACC_TOL = 0.12


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--slug", required=True)
    p.add_argument("--config", default="4096")
    p.add_argument("--tasks", default="niah_multikey_3,vt,fwe,qa_1")
    p.add_argument("--budgets", default="0.0625,0.125,0.25,0.375,0.5,0.625,0.75,0.875")
    p.add_argument("--max_examples", type=int, default=100)
    p.add_argument("--max_new", type=int, default=128)
    p.add_argument("--obs_window", type=int, default=32)
    p.add_argument("--n_sink", type=int, default=4)
    p.add_argument("--top_k", type=int, default=32)
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--floor_alpha", type=float, default=0.5)
    p.add_argument("--device_map", default=None,
                   help="e.g. 'auto' to shard a large model across visible GPUs")
    p.add_argument("--two_pass", action="store_true",
                   help="fallback only: prefill without attentions, then "
                        "re-forward the last obs_window queries. It changes "
                        "what the scorer sees, which is the W9 degeneracy, so "
                        "prefer sharding with --device_map auto.")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    patch_families()
    device = "cuda"
    tasks = args.tasks.split(",")
    budgets = [float(b) for b in args.budgets.split(",")]

    print(f"[load] {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    kw = dict(dtype=torch.bfloat16, attn_implementation="eager")
    if args.device_map:
        kw["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.model, **kw)
    if not args.device_map:
        model = model.to(device)
    model.eval()
    # With device_map the embedding layer may not sit on cuda:0, so feed inputs
    # to wherever the model expects them.
    device = str(getattr(model, "device", device))
    print(f"[device] {device} device_map={args.device_map} "
          f"two_pass={args.two_pass}", flush=True)

    print(f"[data] simonjegou/ruler config={args.config}", flush=True)
    ds = load_dataset("simonjegou/ruler", args.config, split="test")
    rows = []
    for task in tasks:
        sub = ds.filter(lambda r: r["task"] == task)
        for i in range(min(args.max_examples, len(sub))):
            rows.append(sub[i])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t0 = time.time()
    n_done = 0
    with open(args.out, "w") as f:
        for ex_i, ex in enumerate(rows):
            prompt = build_prompt(tokenizer, ex)
            # add_special_tokens=False, matching gated_eviction.py:538. The chat
            # template already emits BOS; letting the tokenizer add another gives
            # a double-BOS. On Llama that shifted T by +1 on every input and
            # drifted D by up to 4e-3. Qwen/Mistral templates do not emit BOS,
            # so the bug was invisible until the Llama control.
            ids = tokenizer(prompt, return_tensors="pt",
                            add_special_tokens=False).input_ids.to(device)
            post_keys, values, scores, gscore, drop, per_layer, T = prefill(
                model, ids, args.obs_window, args.top_k, device,
                two_pass=args.two_pass)
            gate_open = drop >= args.tau
            # Canonical clamp (gated_eviction.py:549-550): take the LARGER of
            # the dataset value and --max_new. RULER gives vt only 30 tokens,
            # but its answer_prefix is ~80 chars, so the model spends the whole
            # budget restating the prefix and never emits the 5 variables.
            # Dropping this clamp put vt full-cache accuracy at 0.000 vs the
            # released 1.000 while the other three tasks matched exactly.
            max_new = int(ex.get("max_new_tokens") or args.max_new)
            max_new = max(max_new, args.max_new)

            # Full-cache reference, once per input.
            pred_full = generate(model, tokenizer, ids, post_keys, values,
                                 None, max_new, device)
            ok_full = is_correct(pred_full, ex["answer"], task=ex["task"])

            for b in budgets:
                rec = {"id": ex_i, "task": ex["task"], "budget": b, "T": T,
                       "drop": drop, "gate_open": bool(gate_open),
                       "correct_full": bool(ok_full)}
                for alloc in ("shared", "adakv"):
                    masks, kept = [], 0.0
                    for li in range(len(post_keys)):
                        # The shared arm is the paper's plain baseline: ONE
                        # global [T] score (layer-averaged) drives an identical
                        # mask in every layer and head. The Ada-KV arm uses the
                        # per-(layer, kv-head) scores. Feeding the shared arm
                        # per-layer scores would give it adaptivity the
                        # baseline does not have.
                        n_kv = scores[li].shape[0]
                        sc = (gscore.unsqueeze(0).expand(n_kv, -1)
                              if alloc == "shared" else scores[li])
                        keep, mk = build_keep_mask(
                            sc, b, args.obs_window, args.n_sink,
                            alloc, args.floor_alpha, device)
                        masks.append(keep)
                        kept += mk
                    kept /= len(post_keys)
                    pred = generate(model, tokenizer, ids, post_keys, values,
                                    masks, max_new, device)
                    ok = is_correct(pred, ex["answer"], task=ex["task"])
                    rec[f"correct_plain_{alloc}"] = bool(ok)
                    rec[f"n_kept_{alloc}"] = kept
                    # Gated arm: evict only when the gate opens, else full cache.
                    rec[f"correct_gated_{alloc}"] = bool(ok) if gate_open else bool(ok_full)
                rec["pred_full"] = pred_full[:120]
                f.write(json.dumps(rec) + "\n")
                f.flush()
            n_done += 1
            del post_keys, values, scores
            torch.cuda.empty_cache()
            if n_done % 5 == 0:
                el = time.time() - t0
                eta = (len(rows) - n_done) / (n_done / el)
                print(f"  [{n_done}/{len(rows)}] task={ex['task']} T={T} "
                      f"drop={drop:.4f} open={gate_open} "
                      f"elapsed={el:.0f}s eta={eta:.0f}s", flush=True)
    print(f"[done] {args.out} in {time.time() - t0:.0f}s", flush=True)

    # Guard: the full-cache arm must match the released full-cache accuracy.
    # A decode-loop bug once flattened this from 0.65 to 0.35 while leaving D
    # exactly reproducible, so gate fidelity alone does not prove a run sound.
    if RELEASED_FULL_ACC.get(args.slug):
        import collections
        seen, acc = {}, collections.defaultdict(list)
        for line in open(args.out):
            r = json.loads(line)
            if (r["task"], r["id"]) not in seen:
                seen[(r["task"], r["id"])] = True
                acc[r["task"]].append(bool(r["correct_full"]))
        bad = []
        for task, want in RELEASED_FULL_ACC[args.slug].items():
            got = acc.get(task)
            if not got:
                continue
            m = sum(got) / len(got)
            if abs(m - want) > FULL_ACC_TOL:
                bad.append(f"{task}: got {m:.3f}, released {want:.3f} (n={len(got)})")
        if bad:
            print("FULL-CACHE ACCURACY CHECK FAILED - do not use this run:",
                  flush=True)
            for b in bad:
                print("  " + b, flush=True)
            raise SystemExit(1)
        print("[check] full-cache accuracy matches the released logs", flush=True)

    # Guard 2: the shared arm IS the paper's plain baseline, so it must
    # reproduce the released plain accuracies on NIAH-MK3.
    if RELEASED_PLAIN_MK3.get(args.slug):
        import collections
        acc = collections.defaultdict(list)
        for line in open(args.out):
            r = json.loads(line)
            if r["task"] == "niah_multikey_3":
                acc[round(r["budget"], 4)].append(bool(r["correct_plain_shared"]))
        bad = []
        for b, want in RELEASED_PLAIN_MK3[args.slug].items():
            got = acc.get(round(b, 4))
            if not got:
                continue
            m = sum(got) / len(got)
            if abs(m - want) > PLAIN_ACC_TOL:
                bad.append(f"b={b}: got {m:.3f}, released {want:.3f} (n={len(got)})")
        if bad:
            print("PLAIN-ARM ACCURACY CHECK FAILED - do not use this run:",
                  flush=True)
            for b in bad:
                print("  " + b, flush=True)
            raise SystemExit(1)
        print("[check] shared/plain arm matches the released plain accuracies",
              flush=True)


if __name__ == "__main__":
    main()