"""Faithful ManifoldKV (arXiv 2602.08343) + Ada-KV (arXiv 2407.11550) reproduction.

Goal: test whether a *faithful* geometry-based scorer (ManifoldKV: L2 distance to
the per-head key centroid) integrated with Ada-KV per-head adaptive budget
allocation rescues RULER niah_multikey_3 (MK3), where attention-score evictors
(SnapKV) collapse under compression.

Method (from the paper):
  ManifoldKV score:  s_i = || k_i - mu ||_2,  mu = (1/N) sum_i k_i   (Algorithm 1)
    - per (layer, kv-head) centroid; keep the top-scoring (outlier) tokens.
  Ada-KV integration: within each layer, the fixed KV budget is *reallocated*
    across kv-heads (head-wise adaptive), instead of a uniform per-head top-k.
    Real Ada-KV: per-head safeguard floor (floor_alpha=0.5 of the uniform budget)
    + global top-k over the remaining flattened per-head scores.

Faithful-simulation design (documented deviation):
  Ada-KV yields *ragged* per-head keep counts, which HF DynamicCache cannot store
  rectangularly. Instead of physically pruning (which would need padding /
  over-allocation), we keep the FULL cache and *mask attention per kv-head*: each
  kv-head only attends to its own keep-set (sink + recency + its Ada-KV-allocated
  middle tokens). This reproduces EXACTLY the logits a ragged compressed cache
  would produce (kept keys retain their original RoPE rotation; evicted keys get
  -inf attention), with zero over-allocation and no position surgery. It measures
  ACCURACY under a given logical budget (not wall-clock memory savings).

Scorers:
  manifoldkv       : L2 to centroid, POST-RoPE keys (as stored in cache)
  manifoldkv_pre   : L2 to centroid, PRE-RoPE keys (k_proj(hidden), no rotary)
  keydiff          : negative cosine to centroid, POST-RoPE (paper's baseline)
  snapkv           : attention mass from last obs_window queries (control)

Allocation:  uniform | adakv

Usage:
  python manifoldkv_adakv.py --config 4096 --tasks niah_multikey_3 \
    --scorers manifoldkv,manifoldkv_pre,keydiff,snapkv \
    --allocs uniform,adakv --budgets 1.0,0.5,0.25,0.125,0.0625 \
    --max_examples 50 --gpu 3 --out results/manifoldkv_faithful_mk3.jsonl
"""
import argparse
import json
import math
import os
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
import transformers.models.qwen2.modeling_qwen2 as qwen2_mod

# ---------------------------------------------------------------------------
# Per-kv-head attention masking (faithful eviction simulation)
# ---------------------------------------------------------------------------
_STATE = {"active": False, "head_mask": None}  # head_mask: list[L] of [n_kv, T] bool keep
_orig_eager = qwen2_mod.eager_attention_forward


def patched_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key_states = qwen2_mod.repeat_kv(key, module.num_key_value_groups)
    value_states = qwen2_mod.repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[..., : key_states.shape[-2]]
    if _STATE["active"]:
        keep = _STATE["head_mask"][module.layer_idx]  # [n_kv, T] bool
        n_kv, T = keep.shape
        cur_kv = attn_weights.shape[-1]
        n_rep = module.num_key_value_groups
        keep_q = keep.repeat_interleave(n_rep, dim=0)  # [n_q, T]
        if cur_kv > T:  # decoded tokens are always kept (recency)
            pad = torch.ones(keep_q.shape[0], cur_kv - T, dtype=torch.bool, device=keep_q.device)
            keep_q = torch.cat([keep_q, pad], dim=1)
        elif cur_kv < T:
            keep_q = keep_q[:, :cur_kv]
        add = torch.zeros_like(keep_q, dtype=attn_weights.dtype)
        add.masked_fill_(~keep_q, float("-inf"))
        attn_weights = attn_weights + add[None, :, None, :]
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


qwen2_mod.eager_attention_forward = patched_eager
# Also patch the registry used via config (_attn_implementation="eager").
try:
    qwen2_mod.ALL_ATTENTION_FUNCTIONS.register("eager", patched_eager)
except Exception:
    pass


# ---------------------------------------------------------------------------
# Prompt / dataset helpers (mirror gated_eviction.py)
# ---------------------------------------------------------------------------
def build_prompt(tokenizer, ex):
    user_msg = ex["context"] + "\n\n" + ex["question"]
    if ex.get("answer_prefix"):
        user_msg = user_msg + "\n" + ex["answer_prefix"]
    messages = [
        {"role": "system", "content": "Answer the question concisely."},
        {"role": "user", "content": user_msg},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def is_correct(pred, golds, task=None):
    pred_lower = pred.lower()
    if task in ("vt", "fwe", "cwe", "niah_multivalue"):
        return all(g.strip().lower() in pred_lower for g in golds)
    return any(g.strip().lower() in pred_lower for g in golds)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def manifold_scores(keys):
    """keys: [1, n_kv, T, d] -> [n_kv, T] L2 distance to per-head centroid."""
    kf = keys[0].float()                       # [n_kv, T, d]
    centroid = kf.mean(dim=1, keepdim=True)     # [n_kv, 1, d]
    return (kf - centroid).norm(dim=-1)         # [n_kv, T]


def keydiff_scores(keys):
    """Negative cosine to per-head centroid (keep = most diverse = lowest cosine)."""
    kf = keys[0].float()
    centroid = kf.mean(dim=1, keepdim=True)
    cos = torch.nn.functional.cosine_similarity(kf, centroid, dim=-1)  # [n_kv, T]
    return -cos


def compute_scores(scorer, post_keys, pre_keys, snap_scores):
    if scorer == "manifoldkv":
        return manifold_scores(post_keys)
    if scorer == "manifoldkv_pre":
        return manifold_scores(pre_keys)
    if scorer == "keydiff":
        return keydiff_scores(post_keys)
    if scorer == "snapkv":
        return snap_scores  # [n_kv, T]
    raise ValueError(scorer)


# ---------------------------------------------------------------------------
# Keep-set construction (uniform vs Ada-KV)
# ---------------------------------------------------------------------------
def build_keep_mask(scores, budget_ratio, obs_window, n_sink, alloc, floor_alpha, device):
    """scores: [n_kv, T] (higher = keep). Returns [n_kv, T] bool keep + mean kept count."""
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
    sc = scores[:, cand_start:cand_end]  # [n_kv, n_cand]

    if alloc == "uniform":
        idx = torch.topk(sc, mid_per_head, dim=1).indices  # [n_kv, mid]
        for h in range(n_kv):
            keep[h, cand_start + idx[h]] = True
    elif alloc == "adakv":
        M_total = mid_per_head * n_kv
        floor_mid = int(math.floor(floor_alpha * mid_per_head))
        floor_mid = min(floor_mid, n_cand)
        work = sc.clone()
        if floor_mid > 0:
            fidx = torch.topk(work, floor_mid, dim=1).indices  # [n_kv, floor]
            for h in range(n_kv):
                keep[h, cand_start + fidx[h]] = True
                work[h, fidx[h]] = float("-inf")  # remove from global pool
        remaining = M_total - floor_mid * n_kv
        if remaining > 0:
            flat = work.reshape(-1)
            remaining = min(remaining, int(torch.isfinite(flat).sum().item()))
            if remaining > 0:
                gidx = torch.topk(flat, remaining).indices
                head_idx = gidx // n_cand
                cand_idx = gidx % n_cand
                keep[head_idx, cand_start + cand_idx] = True
    else:
        raise ValueError(alloc)
    return keep, float(keep.sum(dim=1).float().mean())


# ---------------------------------------------------------------------------
# Prefill + scoring
# ---------------------------------------------------------------------------
def prefill(model, ids, need_snapkv, obs_window, device):
    """Returns per-layer post_keys, pre_keys, values, snap_scores (or None), and past clone."""
    _STATE["active"] = False
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True, output_hidden_states=True, return_dict=True)
    past = out.past_key_values
    post_keys = [layer.keys.clone() for layer in past.layers]
    values = [layer.values.clone() for layer in past.layers]
    hidden = out.hidden_states  # tuple L+1, hidden[i] = input to layer i
    T = ids.shape[1]

    # pre-RoPE keys: k_proj(hidden[i]) reshaped (Qwen2.5 has no q/k norm)
    pre_keys = []
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        hs = hidden[i]  # [1, T, hidden]
        n_kv = attn.config.num_key_value_heads
        d = attn.head_dim
        k = attn.k_proj(hs).view(1, T, n_kv, d).transpose(1, 2).contiguous()  # [1, n_kv, T, d]
        pre_keys.append(k)

    snap = None
    if need_snapkv:
        snap = snapkv_scores(model, ids, post_keys, values, obs_window, device)

    del out
    torch.cuda.empty_cache()
    return post_keys, pre_keys, values, snap, T


def snapkv_scores(model, ids, post_keys, values, obs_window, device):
    """Recompute attention of the last obs_window queries over the full cache to
    get per-kv-head SnapKV scores [L][n_kv, T]. Uses output_attentions (eager)."""
    T = ids.shape[1]
    n_kv = model.config.num_key_value_heads
    n_q = model.config.num_attention_heads
    n_rep = n_q // n_kv

    past_short = DynamicCache()
    for i, (k, v) in enumerate(zip(post_keys, values)):
        past_short.update(k[:, :, : T - obs_window, :].clone(),
                          v[:, :, : T - obs_window, :].clone(), i)
    last_ids = ids[:, -obs_window:]
    pos = torch.arange(T - obs_window, T, device=device).unsqueeze(0)
    cache_pos = torch.arange(T - obs_window, T, device=device)
    _STATE["active"] = False
    with torch.no_grad():
        out = model(input_ids=last_ids, past_key_values=past_short, position_ids=pos,
                    cache_position=cache_pos, output_attentions=True, use_cache=False,
                    return_dict=True)
    scores = []
    for a in out.attentions:  # [1, n_q, obs_window, T]
        s = a[0].float().sum(dim=1)          # [n_q, T]  sum over query positions
        s = s.view(n_kv, n_rep, T).mean(dim=1)  # [n_kv, T] pool q-heads within group
        scores.append(s)
    del out, past_short
    torch.cuda.empty_cache()
    return scores


# ---------------------------------------------------------------------------
# Generation with masked full cache
# ---------------------------------------------------------------------------
def generate(model, tokenizer, ids, post_keys, values, head_masks, max_new, device):
    """Full (unpruned) cache; per-kv-head keep masks applied via patched attention.

    Faithful eviction semantics: the prompt was prefilled UNMASKED (cache intact).
    Compression happens after prefill; the first and all subsequent decoded tokens
    attend only to the (masked) compressed cache. We re-feed the last prompt token
    against the masked cache to produce the first logits (as in gated_eviction.py),
    so no answer token benefits from full-prompt attention.
    """
    past = DynamicCache()
    for i, (k, v) in enumerate(zip(post_keys, values)):
        past.update(k.clone(), v.clone(), i)
    T = ids.shape[1]
    _STATE["head_mask"] = head_masks
    _STATE["active"] = head_masks is not None

    out_ids = []
    next_token = ids[:, -1:].clone()
    next_pos = torch.tensor([[T - 1]], dtype=torch.long, device=device)
    cache_pos = torch.tensor([T], dtype=torch.long, device=device)
    for step in range(max_new):
        with torch.no_grad():
            o = model(input_ids=next_token, past_key_values=past, position_ids=next_pos,
                      cache_position=cache_pos, use_cache=True, return_dict=True)
        past = o.past_key_values
        nid = int(o.logits[:, -1, :].argmax(dim=-1).item())
        out_ids.append(nid)
        if nid == tokenizer.eos_token_id:
            break
        next_token = torch.tensor([[nid]], dtype=torch.long, device=device)
        next_pos = next_pos + 1
        cache_pos = cache_pos + 1
    _STATE["active"] = False
    _STATE["head_mask"] = None
    return tokenizer.decode(out_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="4096")
    p.add_argument("--tasks", default="niah_multikey_3")
    p.add_argument("--scorers", default="manifoldkv,manifoldkv_pre,keydiff,snapkv")
    p.add_argument("--allocs", default="uniform,adakv")
    p.add_argument("--budgets", default="1.0,0.5,0.25,0.125,0.0625")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--max_examples", type=int, default=50)
    p.add_argument("--max_new", type=int, default=64)
    p.add_argument("--obs_window", type=int, default=32)
    p.add_argument("--n_sink", type=int, default=4)
    p.add_argument("--floor_alpha", type=float, default=0.5)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    budgets = [float(b) for b in args.budgets.split(",")]
    tasks = args.tasks.split(",")
    scorers = args.scorers.split(",")
    allocs = args.allocs.split(",")
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"loading RULER {args.config}")
    ds = load_dataset("simonjegou/ruler", args.config, split="test")
    rows = []
    for task in tasks:
        sub = ds.filter(lambda r: r["task"] == task)
        sub = sub.select(range(min(args.max_examples, len(sub))))
        rows.extend(list(sub))
    print(f"loaded {len(rows)} rows across {tasks}")

    print(f"loading {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager", trust_remote_code=True,
    ).to(device).eval()

    need_snapkv = "snapkv" in scorers
    f = open(args.out, "w")
    t0 = time.time()
    for ex_i, ex in enumerate(rows):
        prompt = build_prompt(tokenizer, ex)
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        T = ids.shape[1]
        post_keys, pre_keys, values, snap, T = prefill(model, ids, need_snapkv, args.obs_window, device)
        max_new = max(int(ex.get("max_new_tokens") or args.max_new), args.max_new)

        for scorer in scorers:
            # Precompute per-layer [n_kv, T] scores (higher = keep) once.
            scores_per_layer = []
            for li in range(len(post_keys)):
                if scorer == "snapkv":
                    scores_per_layer.append(snap[li])
                elif scorer == "manifoldkv":
                    scores_per_layer.append(manifold_scores(post_keys[li]))
                elif scorer == "manifoldkv_pre":
                    scores_per_layer.append(manifold_scores(pre_keys[li]))
                elif scorer == "keydiff":
                    scores_per_layer.append(keydiff_scores(post_keys[li]))
                else:
                    raise ValueError(scorer)
            for alloc in allocs:
                for b in budgets:
                    if b >= 1.0:
                        head_masks = None  # full cache
                        mean_kept = float(T)
                        if alloc != allocs[0]:
                            continue  # full-KV identical across allocs; log once
                    else:
                        head_masks = []
                        mk = 0.0
                        for li in range(len(post_keys)):
                            keep, mkl = build_keep_mask(scores_per_layer[li], b, args.obs_window,
                                                        args.n_sink, alloc, args.floor_alpha, device)
                            head_masks.append(keep)
                            mk += mkl
                        mean_kept = mk / len(post_keys)
                    pred = generate(model, tokenizer, ids, post_keys, values, head_masks, max_new, device)
                    ok = is_correct(pred, ex["answer"], task=ex["task"])
                    rec = {
                        "id": ex_i, "task": ex["task"], "scorer": scorer,
                        "alloc": alloc if b < 1.0 else "full", "budget": b, "T": T,
                        "mean_kept": mean_kept, "eff_budget": mean_kept / T,
                        "gold": ex["answer"], "pred": pred[:160], "correct": bool(ok),
                    }
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
        del post_keys, pre_keys, values, snap
        torch.cuda.empty_cache()
        if (ex_i + 1) % 5 == 0:
            el = time.time() - t0
            eta = (len(rows) - ex_i - 1) / ((ex_i + 1) / el)
            print(f"  [{ex_i+1}/{len(rows)}] task={ex['task']} T={T} elapsed={el:.0f}s eta={eta:.0f}s")

    f.close()
    print(f"done in {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
