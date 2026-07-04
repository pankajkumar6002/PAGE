"""H1 spot-check: sweep KV budgets on NIAH-MultiKey examples.

For each example, prefill once with output_attentions (eager). Derive a
per-budget keep_mask from last-window attention scores averaged over layers
and heads. Slice the KV cache to keep only those positions, then generate
the answer with explicit position_ids preserving original positions.

H1 question: is there a non-trivial fraction of inputs where some
budget_ratio < 1.0 yields strictly higher accuracy than budget_ratio == 1.0?

Eviction policy: SnapKV-style, single mask shared across layers. The first
n_sink tokens are always kept; the last obs_window tokens are always kept;
among the middle positions, top-k are kept by score.
"""
import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


def build_prompt(tokenizer, ex):
    messages = [
        {"role": "system", "content": "Answer with the integer only, no explanation."},
        {"role": "user", "content": ex["context"] + "\n\n" + ex["question"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


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


def slice_cache(past, keep_idx):
    """Slice every layer's K and V along the sequence dimension.

    `past`: DynamicCache from prefill (length T).
    `keep_idx`: 1D LongTensor of positions to retain.
    Returns: new DynamicCache of length len(keep_idx).
    """
    new = DynamicCache()
    L = len(past.key_cache)
    for i in range(L):
        k = past.key_cache[i].index_select(2, keep_idx)
        v = past.value_cache[i].index_select(2, keep_idx)
        new.update(k, v, i)
    return new


def generate_with_evicted_cache(model, tokenizer, ids, score, budget_ratio, max_new,
                                obs_window, n_sink, device):
    """Slice the prefilled cache to top-budget positions and generate greedily."""
    T = ids.shape[1]
    keep_mask = derive_keep_mask(score, T, budget_ratio, obs_window, n_sink, device)
    keep_idx = keep_mask.nonzero(as_tuple=True)[0]
    n_kept = int(keep_idx.shape[0])

    # Re-prefill once (we already prefilled outside; re-doing here is wasteful but
    # we need the prefill_out under the same model state for each budget. For the
    # budget==1.0 branch we want the trivial unmodified pass.) To avoid duplicate
    # prefills, the caller passes the prefilled `past` as `score`'s sibling; here
    # we recompute past from scratch only for the budget<1.0 case.
    raise NotImplementedError  # see main()


def generate_from_past(model, tokenizer, past, ids, max_new, device, original_T):
    """Greedy decode from a (possibly sliced) past KV cache, with explicit position_ids
    so that the next-token position is `original_T` regardless of past length.
    """
    # First new token: re-attend the LAST prompt token against the (sliced) past.
    # This gives the correctly-masked first-token logit (unlike using prefill's
    # last_logits directly, which was computed with the unsliced cache).
    last_prompt_id = ids[:, -1:].clone()  # [1, 1]

    # Move past_key_values' "seen tokens" counter so cache_position aligns.
    # We pass position_ids explicitly and let the cache handle the rest.
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
        new_token_id = int(out.logits[:, -1, :].argmax(dim=-1).item())
        # The first iteration re-attended the last prompt token; its logit is the
        # first new token's prediction.
        out_ids.append(new_token_id)
        if new_token_id == tokenizer.eos_token_id:
            break
        next_token = torch.tensor([[new_token_id]], dtype=torch.long, device=device)
        next_position = next_position + 1
        cache_position = cache_position + 1

    return tokenizer.decode(out_ids, skip_special_tokens=True).strip()


def correct(pred, gold):
    return gold.strip() in pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/home/smlab/projects/eff-nn/experiments/data/niah_multikey.jsonl")
    p.add_argument("--out", default="/home/smlab/projects/eff-nn/experiments/results/h1_sweep.jsonl")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--budgets", default="1.0,0.5,0.25,0.125,0.0625")
    p.add_argument("--max_examples", type=int, default=100)
    p.add_argument("--max_new", type=int, default=16)
    p.add_argument("--obs_window", type=int, default=32)
    p.add_argument("--n_sink", type=int, default=4)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--eviction", default="snapkv", choices=["snapkv", "random"],
                   help="snapkv: score by last-window attention; random: uniform random scores")
    p.add_argument("--seed", type=int, default=20260602)
    args = p.parse_args()

    budgets = [float(b) for b in args.budgets.split(",")]
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"loading {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device).eval()

    examples = []
    with open(args.data) as f:
        for line in f:
            examples.append(json.loads(line))
    examples = examples[: args.max_examples]
    print(f"loaded {len(examples)} examples, sweeping budgets {budgets}")

    results_f = open(args.out, "w")
    t0 = time.time()
    for ex_i, ex in enumerate(examples):
        prompt = build_prompt(tokenizer, ex)
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        T = ids.shape[1]

        need_attn = (args.eviction == "snapkv")
        with torch.no_grad():
            prefill = model(
                input_ids=ids,
                output_attentions=need_attn,
                use_cache=True,
                return_dict=True,
            )
        past_full = prefill.past_key_values
        if need_attn:
            attn_stack = torch.stack([
                a[0, :, -args.obs_window:, :].mean(dim=(0, 1)) for a in prefill.attentions
            ])  # [L, T]
            score = attn_stack.mean(dim=0)  # [T]
            del attn_stack
        else:
            g = torch.Generator(device=device).manual_seed(args.seed + ex_i)
            score = torch.rand(T, generator=g, device=device)

        full_keys = [layer.keys.clone() for layer in past_full.layers]
        full_values = [layer.values.clone() for layer in past_full.layers]
        del prefill, past_full
        torch.cuda.empty_cache()

        for b in budgets:
            keep_mask = derive_keep_mask(score, T, b, args.obs_window, args.n_sink, device)
            keep_idx = keep_mask.nonzero(as_tuple=True)[0]
            n_kept = int(keep_idx.shape[0])

            past_b = DynamicCache()
            for i, (k, v) in enumerate(zip(full_keys, full_values)):
                past_b.update(
                    k.index_select(2, keep_idx),
                    v.index_select(2, keep_idx),
                    i,
                )

            pred = generate_from_past(
                model, tokenizer, past_b, ids, args.max_new, device, original_T=T,
            )
            ok = correct(pred, ex["answer"])
            rec = {
                "id": ex["id"],
                "budget": b,
                "T": T,
                "n_kept": n_kept,
                "gold": ex["answer"],
                "pred": pred,
                "correct": bool(ok),
            }
            results_f.write(json.dumps(rec) + "\n")
            results_f.flush()
            del past_b
            torch.cuda.empty_cache()

        del full_keys, full_values
        torch.cuda.empty_cache()

        if (ex_i + 1) % 5 == 0:
            elapsed = time.time() - t0
            rate = (ex_i + 1) / elapsed
            eta = (len(examples) - ex_i - 1) / rate
            print(f"  [{ex_i+1}/{len(examples)}]  T={T}  elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    results_f.close()
    print(f"done in {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
