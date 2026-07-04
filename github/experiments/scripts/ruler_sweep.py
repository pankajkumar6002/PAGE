"""H1 sweep on the actual RULER benchmark (simonjegou/ruler on HF).

Loads RULER at the requested context-length config, filters by task name,
and runs the same KV-budget sweep as h1_sweep.py but on real benchmark data.
"""
import argparse
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


def prefill_and_score(model, ids, obs_window, device, two_pass):
    """Returns (full K/V tensors, score [T]).

    one_pass: forward with output_attentions=True (memory O(L*H*T^2)).
    two_pass: forward without attentions (cheap), then re-forward last
        obs_window tokens against a sliced past with output_attentions=True.
        Memory for the score pass is O(L*H*obs_window*T) — tiny.
    """
    T = ids.shape[1]
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
        score = torch.stack([
            a[0, :, -obs_window:, :].mean(dim=(0, 1)) for a in out.attentions
        ]).mean(dim=0)
        del out, past
        torch.cuda.empty_cache()
        return full_keys, full_values, score

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

    # Pass-2 needs output_attentions; force eager just for this small call
    # (obs_window queries x T keys), then restore the loaded implementation.
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
    score = torch.stack([
        a[0].mean(dim=(0, 1)) for a in scoring.attentions
    ]).mean(dim=0)
    del scoring, past_short
    torch.cuda.empty_cache()
    return full_keys, full_values, score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="4096", choices=["4096", "8192", "16384", "32768", "65536"])
    p.add_argument("--tasks", default="niah_multikey_3", help="comma-separated RULER task names")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--budgets", default="1.0,0.875,0.75,0.625,0.5,0.375,0.25,0.1875,0.125,0.0625")
    p.add_argument("--max_examples", type=int, default=200, help="per task")
    p.add_argument("--max_new", type=int, default=128)
    p.add_argument("--obs_window", type=int, default=32)
    p.add_argument("--n_sink", type=int, default=4)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--eviction", default="snapkv", choices=["snapkv", "random"])
    p.add_argument("--two_pass", action="store_true",
                   help="Use two-pass scoring (required for >=16K context to avoid OOM)")
    p.add_argument("--seed", type=int, default=20260603)
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
        sub = ds.filter(lambda r: r["task"] == task)
        sub = sub.select(range(min(args.max_examples, len(sub))))
        for r in sub:
            rows.append(r)
    print(f"loaded {len(rows)} rows across tasks {tasks}")

    print(f"loading {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        # sdpa for two-pass: eager pass-1 materializes the full T x T
        # attention matrix and OOMs at 16K (same fix as gated_eviction.py).
        attn_implementation="sdpa" if args.two_pass else "eager",
    ).to(device).eval()

    results_f = open(args.out, "w")
    t0 = time.time()
    for ex_i, ex in enumerate(rows):
        prompt = build_prompt(tokenizer, ex)
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        T = ids.shape[1]

        if args.eviction == "snapkv":
            full_keys, full_values, score = prefill_and_score(
                model, ids, args.obs_window, device, two_pass=args.two_pass,
            )
        else:
            with torch.no_grad():
                out = model(
                    input_ids=ids,
                    output_attentions=False,
                    use_cache=True,
                    return_dict=True,
                )
            past_full = out.past_key_values
            full_keys = [layer.keys.clone() for layer in past_full.layers]
            full_values = [layer.values.clone() for layer in past_full.layers]
            del out, past_full
            torch.cuda.empty_cache()
            g = torch.Generator(device=device).manual_seed(args.seed + ex_i)
            score = torch.rand(T, generator=g, device=device)

        max_new = int(ex.get("max_new_tokens") or args.max_new)
        max_new = max(max_new, args.max_new)

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

            pred = generate_from_past(model, tokenizer, past_b, ids, max_new, device, original_T=T)
            ok = is_correct(pred, ex["answer"], task=ex["task"])
            rec = {
                "id": ex_i,
                "task": ex["task"],
                "budget": b,
                "T": T,
                "n_kept": n_kept,
                "gold": ex["answer"],
                "pred": pred[:200],
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
            eta = (len(rows) - ex_i - 1) / rate
            print(f"  [{ex_i+1}/{len(rows)}]  task={ex['task']}  T={T}  elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    results_f.close()
    print(f"done in {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
