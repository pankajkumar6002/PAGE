"""Gated KV-cache eviction on LongBench subtasks.

Uses the same gating algorithm as gated_eviction.py but loads from
Xnhyacinth/LongBench instead of simonjegou/ruler. Each LongBench config
is loaded separately.

Subtasks evaluated by default (chosen for tractable context + clear metric):
  - qasper (single-doc QA, F1 answer)
  - multifieldqa_en (single-doc QA, short context)
  - trec (few-shot classification)
  - triviaqa (multi-answer QA)

Metric: any-in substring match (proxy for F1 / accuracy).
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
        {"role": "system", "content": "Answer the question based on the context. Be concise."},
        {"role": "user", "content": user_msg},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


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


def prefill_and_score_with_agreement(model, ids, obs_window, top_k, device, two_pass):
    T = ids.shape[1]
    loaded_impl = getattr(model.config, "_attn_implementation", "eager")

    def switch_layers(impl):
        for layer in model.model.layers:
            try:
                layer.self_attn.config._attn_implementation = impl
            except AttributeError:
                pass

    if not two_pass:
        with torch.no_grad():
            out = model(input_ids=ids, output_attentions=True, use_cache=True, return_dict=True)
        past = out.past_key_values
        full_keys = [layer.keys.clone() for layer in past.layers]
        full_values = [layer.values.clone() for layer in past.layers]
        score = torch.stack([a[0, :, -obs_window:, :].mean(dim=(0, 1)) for a in out.attentions]).mean(dim=0)
        drop, per_layer = compute_drop_from_attentions(out.attentions, obs_window, top_k)
        del out, past
        torch.cuda.empty_cache()
        return full_keys, full_values, score, drop, per_layer

    with torch.no_grad():
        prefill = model(input_ids=ids, output_attentions=False, use_cache=True, return_dict=True)
    past = prefill.past_key_values
    full_keys = [layer.keys.clone() for layer in past.layers]
    full_values = [layer.values.clone() for layer in past.layers]
    del prefill, past
    torch.cuda.empty_cache()

    past_short = DynamicCache()
    for i, (k, v) in enumerate(zip(full_keys, full_values)):
        past_short.update(k[:, :, : T - obs_window, :].clone(), v[:, :, : T - obs_window, :].clone(), i)
    last_ids = ids[:, -obs_window:]
    pos_ids = torch.arange(T - obs_window, T, device=device).unsqueeze(0)
    cache_pos = torch.arange(T - obs_window, T, device=device)

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
    score = torch.stack([a[0].mean(dim=(0, 1)) for a in scoring.attentions]).mean(dim=0)
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


def is_correct(pred, golds):
    """any-in match: any gold answer is a substring of pred. LongBench uses
    F1 for QA tasks; substring match approximates token recall."""
    pred_lower = pred.lower()
    return any(g.strip().lower() in pred_lower for g in golds if g.strip())


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default="qasper,multifieldqa_en,trec,triviaqa",
                   help="LongBench subtask names (separate configs)")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--budgets", default="1.0,0.5,0.25,0.125")
    p.add_argument("--max_examples", type=int, default=50)
    p.add_argument("--max_new", type=int, default=128)
    p.add_argument("--obs_window", type=int, default=32)
    p.add_argument("--n_sink", type=int, default=4)
    p.add_argument("--top_k", type=int, default=32)
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--two_pass", action="store_true")
    p.add_argument("--attn_impl", default="sdpa", choices=["eager", "sdpa"])
    p.add_argument("--max_context_tokens", type=int, default=24000,
                   help="Skip examples whose tokenized prompt exceeds this length.")
    args = p.parse_args()

    budgets = [float(b) for b in args.budgets.split(",")]
    tasks = args.tasks.split(",")
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"loading {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
    ).to(device).eval()

    # Load all subtask rows
    rows = []
    for task in tasks:
        print(f"loading LongBench:{task}")
        ds = load_dataset("Xnhyacinth/LongBench", task, split="test")
        ds = ds.select(range(min(args.max_examples, len(ds))))
        for r in ds:
            r = dict(r)
            r["_task"] = task
            rows.append(r)
    print(f"total {len(rows)} examples")
    print(f"tau = {args.tau}")

    results_f = open(args.out, "w")
    t0 = time.time()
    n_skipped = 0
    for ex_i, ex in enumerate(rows):
        prompt = build_prompt(tokenizer, ex)
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, truncation=False).input_ids.to(device)
        T = ids.shape[1]
        if T > args.max_context_tokens:
            n_skipped += 1
            continue

        try:
            full_keys, full_values, score, drop, _ = prefill_and_score_with_agreement(
                model, ids, args.obs_window, args.top_k, device, two_pass=args.two_pass,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            n_skipped += 1
            continue

        gate_open = drop >= args.tau
        max_new = int(ex.get("max_new_tokens") or args.max_new)
        max_new = max(max_new, args.max_new)

        for b in budgets:
            pred_plain, n_kept_plain = evaluate_one(
                model, tokenizer, ids, full_keys, full_values, score, T,
                b, max_new, device, args.obs_window, args.n_sink, do_evict=True,
            )
            pred_gated, n_kept_gated = evaluate_one(
                model, tokenizer, ids, full_keys, full_values, score, T,
                b, max_new, device, args.obs_window, args.n_sink, do_evict=bool(gate_open),
            )

            ok_plain = is_correct(pred_plain, ex["answers"])
            ok_gated = is_correct(pred_gated, ex["answers"])

            rec = {
                "id": ex_i,
                "task": ex["_task"],
                "budget": b,
                "T": T,
                "drop": float(drop),
                "gate_open": bool(gate_open),
                "n_kept_plain": n_kept_plain,
                "n_kept_gated": n_kept_gated,
                "answers": ex["answers"],
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
            eta = (len(rows) - ex_i - 1) * elapsed / max(1, ex_i + 1)
            print(f"  [{ex_i+1}/{len(rows)}]  task={ex['_task']}  T={T}  drop={drop:+.4f}  "
                  f"gate_open={gate_open}  elapsed={elapsed:.0f}s  eta={eta:.0f}s  skipped={n_skipped}")

    results_f.close()
    print(f"done in {time.time()-t0:.0f}s -> {args.out}  (skipped {n_skipped} too-long)")


if __name__ == "__main__":
    main()
