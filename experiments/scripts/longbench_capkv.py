"""Gated vs plain CapKV head-to-head on a LongBench subtask.

CapKV reference: arXiv:2604.25975 ("Rethinking KV Cache Eviction via a Unified
Information-Theoretic Objective", Yang et al. 2026). No official public code
release exists at the time of writing, so we implement the core scoring rule
inline as a faithful reproduction proxy (`--score_policy capkv_proxy`):

  - For each layer, capture per-token queries q_t and keys k_i, values v_i.
  - mu_q = mean over the last `obs_window` query positions (the "historical
    queries" in CapKV's online formulation).
  - per-token weight     w_i = exp(<k_i, mu_q> * tau)                (Eq. 8)
  - per-token output dir u_i = v_i                                   (Sec 3.5)
  - capacity matrix      A   = I + sum_i w_i u_i u_i^T               (Eq. 6)
  - leverage score       s_i = w_i u_i^T A^{-1} u_i                  (Eq. 7)
  - retain tokens with the largest s_i.

We compute s per (layer, kv_head) and average across heads, then average
across layers, to produce a single [T] score that is used uniformly by the
layer-uniform keep-mask path. CapKV's published table uses tau=5 as the
default; we use that here.

Gating wrapper is the same as longbench_gating.py: compute the head-agreement
early-vs-late drop; if drop >= tau_gate, apply CapKV-proxy eviction at budget
b; otherwise keep full KV.

Outputs one record per (example, budget) to a jsonl with plain vs gated
predictions, the n_kept counts, and correctness under any-in substring match.
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


# ---------------------------------------------------------------------------
# Prompt + simple metric (same as longbench_gating.py).

def build_prompt(tokenizer, ex):
    user_msg = ex["context"] + "\n\n" + ex["question"]
    if ex.get("answer_prefix"):
        user_msg = user_msg + "\n" + ex["answer_prefix"]
    messages = [
        {"role": "system", "content": "Answer the question based on the context. Be concise."},
        {"role": "user", "content": user_msg},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def is_correct(pred, golds):
    pred_lower = pred.lower()
    return any(g.strip().lower() in pred_lower for g in golds if g.strip())


# ---------------------------------------------------------------------------
# Head-agreement gating signal.

def jaccard(s1, s2):
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return inter / union if union else 0.0


def head_agreement_layer(attn_layer, top_k):
    H, W, T = attn_layer.shape
    head_mean = attn_layer.mean(dim=1)
    top_idx = head_mean.topk(min(top_k, T), dim=-1).indices
    sets = [set(top_idx[h].cpu().tolist()) for h in range(H)]
    pairs = list(itertools.combinations(range(H), 2))
    if not pairs:
        return 0.0
    return sum(jaccard(sets[i], sets[j]) for i, j in pairs) / len(pairs)


def compute_drop_from_attentions(attentions, obs_window, top_k):
    L = len(attentions)
    per_layer = []
    for a in attentions:
        if a.shape[2] != obs_window:
            window = a[0, :, -obs_window:, :]
        else:
            window = a[0]
        per_layer.append(head_agreement_layer(window, top_k))
    third = max(1, L // 3)
    early = sum(per_layer[:third]) / third
    late = sum(per_layer[L - third:]) / third
    return early - late, per_layer


# ---------------------------------------------------------------------------
# Query capture via forward hooks on q_proj.

class QueryCapture:
    """Captures the q_proj output for each layer during the prefill forward."""

    def __init__(self, model):
        self.model = model
        self.layers = model.model.layers
        self.queries = [None] * len(self.layers)
        self._handles = []

    def __enter__(self):
        for i, layer in enumerate(self.layers):
            def mk_hook(idx):
                def _hook(_mod, _inp, out):
                    self.queries[idx] = out.detach()
                return _hook
            self._handles.append(layer.self_attn.q_proj.register_forward_hook(mk_hook(i)))
        return self

    def __exit__(self, *a):
        for h in self._handles:
            h.remove()


# ---------------------------------------------------------------------------
# CapKV scoring (proxy implementation).

def capkv_score_one_layer(keys, values, queries, num_q_heads, num_kv_heads,
                          head_dim, obs_window, tau_capkv):
    """
    keys:    [1, H_kv, T, d]
    values:  [1, H_kv, T, d]
    queries: [1, T, H_q * d]     (q_proj output, as returned by Linear)
    Returns: [T] per-position score, averaged over kv heads (GQA-aware).
    """
    T = keys.shape[2]
    device = keys.device
    # Reshape queries to [1, T, H_q, d]
    queries = queries.view(1, T, num_q_heads, head_dim)

    # Use the last obs_window queries as the empirical historical queries.
    q_hist = queries[:, max(0, T - obs_window):T, :, :]  # [1, W, H_q, d]
    mu_q_per_head = q_hist.mean(dim=1).squeeze(0)         # [H_q, d]

    # GQA grouping: which q heads share a kv head.
    group = num_q_heads // num_kv_heads
    score = torch.zeros(T, device=device, dtype=torch.float32)

    for h in range(num_kv_heads):
        k_h = keys[0, h].float()              # [T, d]
        v_h = values[0, h].float()            # [T, d]
        # mu_q for this kv head = mean over the q heads in its group.
        mu_q = mu_q_per_head[h * group:(h + 1) * group].mean(dim=0).float()  # [d]
        # w_i = exp(<k_i, mu_q> * tau)
        # Stabilize numerically: subtract max before exp.
        align = k_h @ mu_q                    # [T]
        align = align * tau_capkv / (head_dim ** 0.5)  # scale like attention
        align = align - align.max()
        w = torch.exp(align)                  # [T]

        # A = I + sum_i w_i v_i v_i^T   (d x d)
        # Sum can be written as V^T diag(w) V.
        VW = v_h * w.unsqueeze(1)             # [T, d]
        A = v_h.T @ VW                        # [d, d]
        A = A + torch.eye(head_dim, device=device, dtype=A.dtype)

        # Solve A x_i = v_i  for each i, then s_i = w_i * v_i . x_i
        # In batch: X = A^{-1} V^T  -> per-token  s_i = w_i * (v_i^T A^{-1} v_i)
        try:
            A_inv_V = torch.linalg.solve(A, v_h.T)   # [d, T]
        except RuntimeError:
            A_inv_V = torch.linalg.pinv(A) @ v_h.T
        s_per_token = w * (v_h * A_inv_V.T).sum(dim=1)  # [T]
        score = score + s_per_token

    score = score / num_kv_heads
    return score


def derive_keep_mask(score, T, budget_ratio, obs_window, n_sink, device):
    if budget_ratio >= 1.0:
        return torch.ones(T, dtype=torch.bool, device=device)
    budget = max(obs_window + n_sink + 4, int(T * budget_ratio))
    n_middle = max(0, budget - n_sink - obs_window)
    keep = torch.zeros(T, dtype=torch.bool, device=device)
    keep[:n_sink] = True
    keep[T - obs_window:] = True
    cand_start, cand_end = n_sink, T - obs_window
    if n_middle > 0 and cand_end > cand_start:
        cs = score[cand_start:cand_end]
        k = min(n_middle, cs.shape[0])
        topk = torch.topk(cs, k).indices
        keep[cand_start + topk] = True
    return keep


# ---------------------------------------------------------------------------
# Single prefill pass (eager attentions); produces full K/V, agreement drop,
# AND the CapKV-proxy per-position score.

def prefill_and_score(model, ids, obs_window, top_k, tau_capkv, device):
    cfg = model.config
    num_q_heads = cfg.num_attention_heads
    num_kv_heads = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim",
                       cfg.hidden_size // num_q_heads)

    with QueryCapture(model) as qcap:
        with torch.no_grad():
            out = model(input_ids=ids, output_attentions=True, use_cache=True, return_dict=True)

    past = out.past_key_values
    full_keys = [layer.keys.clone() for layer in past.layers]
    full_values = [layer.values.clone() for layer in past.layers]

    # Build snapkv-style score (fallback / sanity), then build capkv_proxy score.
    snap_score = torch.stack(
        [a[0, :, -obs_window:, :].mean(dim=(0, 1)) for a in out.attentions]
    ).mean(dim=0)

    L = len(full_keys)
    per_layer_caps = []
    for li in range(L):
        s = capkv_score_one_layer(
            full_keys[li], full_values[li], qcap.queries[li],
            num_q_heads, num_kv_heads, head_dim, obs_window, tau_capkv,
        )
        per_layer_caps.append(s)
    capkv_score = torch.stack(per_layer_caps).mean(dim=0)

    drop, _ = compute_drop_from_attentions(out.attentions, obs_window, top_k)
    del out, past
    torch.cuda.empty_cache()
    return full_keys, full_values, snap_score, capkv_score, drop


# ---------------------------------------------------------------------------
# Generation from a chosen kept subset.

def generate_from_past(model, tokenizer, past, ids, max_new, device, original_T):
    last_prompt_id = ids[:, -1:].clone()
    n_kept = past.layers[0].keys.shape[2]
    out_ids = []
    next_token = last_prompt_id
    next_position = torch.tensor([[original_T - 1]], dtype=torch.long, device=device)
    cache_position = torch.tensor([n_kept], dtype=torch.long, device=device)
    for _ in range(max_new):
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


def evaluate_one(model, tokenizer, ids, full_keys, full_values, score, T,
                 budget_ratio, max_new, device, obs_window, n_sink, do_evict):
    if do_evict and budget_ratio < 1.0:
        keep_mask = derive_keep_mask(score, T, budget_ratio, obs_window, n_sink, device)
    else:
        keep_mask = torch.ones(T, dtype=torch.bool, device=device)
    keep_idx = keep_mask.nonzero(as_tuple=True)[0]
    n_kept = int(keep_idx.shape[0])
    past_b = DynamicCache()
    for i, (k, v) in enumerate(zip(full_keys, full_values)):
        past_b.update(k.index_select(2, keep_idx), v.index_select(2, keep_idx), i)
    pred = generate_from_past(model, tokenizer, past_b, ids, max_new, device, original_T=T)
    del past_b
    return pred, n_kept


# ---------------------------------------------------------------------------
# Main loop.

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="qasper")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--budget", type=float, default=0.5)
    p.add_argument("--max_examples", type=int, default=50)
    p.add_argument("--max_new", type=int, default=64)
    p.add_argument("--obs_window", type=int, default=32)
    p.add_argument("--n_sink", type=int, default=4)
    p.add_argument("--top_k", type=int, default=32, help="top-k for head agreement")
    p.add_argument("--tau_gate", type=float, default=0.07,
                   help="head-agreement drop threshold for gating wrapper")
    p.add_argument("--tau_capkv", type=float, default=5.0,
                   help="CapKV alignment temperature (paper default = 5)")
    p.add_argument("--max_context_tokens", type=int, default=16000)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"loading {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # eager is required for output_attentions=True
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager",
    ).to(device).eval()

    print(f"loading LongBench:{args.task}")
    ds = load_dataset("Xnhyacinth/LongBench", args.task, split="test")
    ds = ds.select(range(min(args.max_examples, len(ds))))
    print(f"  {len(ds)} examples")
    print(f"  budget={args.budget}  tau_gate={args.tau_gate}  tau_capkv={args.tau_capkv}")

    results_f = open(args.out, "w")
    t0 = time.time()
    n_skipped = 0
    n_done = 0
    n_correct_plain = 0
    n_correct_gated = 0
    n_gate_open = 0
    for ex_i, ex in enumerate(ds):
        prompt = build_prompt(tokenizer, ex)
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, truncation=False).input_ids.to(device)
        T = ids.shape[1]
        if T > args.max_context_tokens:
            n_skipped += 1
            continue

        try:
            full_keys, full_values, _snap, capkv_score, drop = prefill_and_score(
                model, ids, args.obs_window, args.top_k, args.tau_capkv, device,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            n_skipped += 1
            continue
        except Exception as e:
            print(f"  prefill failed on ex {ex_i}: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()
            n_skipped += 1
            continue

        gate_open = bool(drop >= args.tau_gate)
        max_new = max(int(ex.get("max_new_tokens") or args.max_new), args.max_new)

        pred_plain, n_kept_plain = evaluate_one(
            model, tokenizer, ids, full_keys, full_values, capkv_score, T,
            args.budget, max_new, device, args.obs_window, args.n_sink,
            do_evict=True,
        )
        pred_gated, n_kept_gated = evaluate_one(
            model, tokenizer, ids, full_keys, full_values, capkv_score, T,
            args.budget, max_new, device, args.obs_window, args.n_sink,
            do_evict=gate_open,
        )
        ok_plain = is_correct(pred_plain, ex["answers"])
        ok_gated = is_correct(pred_gated, ex["answers"])

        rec = {
            "id": int(ex_i),
            "task": args.task,
            "budget": float(args.budget),
            "T": int(T),
            "drop": float(drop),
            "gate_open": gate_open,
            "n_kept_plain": int(n_kept_plain),
            "n_kept_gated": int(n_kept_gated),
            "answers": list(ex["answers"]),
            "pred_plain": pred_plain[:200],
            "pred_gated": pred_gated[:200],
            "correct_plain": bool(ok_plain),
            "correct_gated": bool(ok_gated),
        }
        results_f.write(json.dumps(rec) + "\n")
        results_f.flush()

        n_done += 1
        n_correct_plain += int(ok_plain)
        n_correct_gated += int(ok_gated)
        n_gate_open += int(gate_open)
        del full_keys, full_values, capkv_score
        torch.cuda.empty_cache()

        if n_done % 5 == 0 or ex_i == len(ds) - 1:
            elapsed = time.time() - t0
            eta = (len(ds) - ex_i - 1) * elapsed / max(1, ex_i + 1)
            print(f"  [{ex_i+1}/{len(ds)}]  T={T}  drop={drop:+.4f}  "
                  f"gate_open={gate_open}  plain={n_correct_plain}/{n_done}  "
                  f"gated={n_correct_gated}/{n_done}  elapsed={elapsed:.0f}s  "
                  f"eta={eta:.0f}s  skipped={n_skipped}")

    results_f.close()
    print(f"done in {time.time()-t0:.0f}s")
    print(f"  N={n_done}  skipped={n_skipped}")
    if n_done > 0:
        print(f"  plain acc = {n_correct_plain / n_done:.4f}")
        print(f"  gated acc = {n_correct_gated / n_done:.4f}")
        print(f"  delta     = {(n_correct_gated - n_correct_plain) / n_done:+.4f}")
        print(f"  gate_open = {n_gate_open}/{n_done}")


if __name__ == "__main__":
    main()
