"""K-interpolation probe: how does the head-agreement drop track distractor density?

Loads synthetic NIAH datasets at K = 1, 2, 4, 8, 16, 32 keys (built by
build_niah.py) and measures mean head-agreement drop (using pass-2 scoring
attention) for each K. The drop is the early-third minus late-third layer-
mean Jaccard top-32 across heads, same definition we use in gated_eviction.py.

Output: a table of (K, mean_T, mean_drop, std_drop) + a TSV ready for
plotting drop vs K.

Strategic motivation: if drop varies smoothly and monotonically with K, the
drop predictor is a continuous measure of dilution headroom — not a binary
"task-family detector" — which neutralizes the "folklore" reviewer objection.
"""
import argparse
import itertools
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


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


def compute_drop(attentions, obs_window, top_k):
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
    return early - late


def build_prompt(tokenizer, ex):
    messages = [
        {"role": "system", "content": "Answer with the integer only, no explanation."},
        {"role": "user", "content": ex["context"] + "\n\n" + ex["question"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", required=True,
                   help="Paths to synthetic NIAH JSONL files (one per K value)")
    p.add_argument("--ks", nargs="+", required=True,
                   help="K values corresponding to the datasets")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--n_per_dataset", type=int, default=30)
    p.add_argument("--obs_window", type=int, default=32)
    p.add_argument("--top_k", type=int, default=32)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--two_pass", action="store_true")
    args = p.parse_args()

    if len(args.datasets) != len(args.ks):
        raise SystemExit("--datasets and --ks must have the same length")

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)

    loaded_impl = "sdpa" if args.two_pass else "eager"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=loaded_impl,
    ).to(device).eval()

    def switch_layers(impl):
        for layer in model.model.layers:
            try:
                layer.self_attn.config._attn_implementation = impl
            except AttributeError:
                pass

    out_f = open(args.out, "w")
    summary = []
    for k, ds_path in zip(args.ks, args.datasets):
        rows = []
        with open(ds_path) as f:
            for line in f:
                rows.append(json.loads(line))
        rows = rows[: args.n_per_dataset]

        drops = []
        Ts = []
        t0 = time.time()
        for ex_i, ex in enumerate(rows):
            prompt = build_prompt(tokenizer, ex)
            ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            T = ids.shape[1]

            if not args.two_pass:
                with torch.no_grad():
                    out = model(input_ids=ids, output_attentions=True, use_cache=False, return_dict=True)
                attentions = out.attentions
            else:
                with torch.no_grad():
                    prefill = model(input_ids=ids, output_attentions=False, use_cache=True, return_dict=True)
                past = prefill.past_key_values
                past_short = DynamicCache()
                for i, layer in enumerate(past.layers):
                    past_short.update(
                        layer.keys[:, :, : T - args.obs_window, :].clone(),
                        layer.values[:, :, : T - args.obs_window, :].clone(),
                        i,
                    )
                last_ids = ids[:, -args.obs_window:]
                pos_ids = torch.arange(T - args.obs_window, T, device=device).unsqueeze(0)
                cache_pos = torch.arange(T - args.obs_window, T, device=device)
                switch_layers("eager")
                try:
                    with torch.no_grad():
                        scoring = model(
                            input_ids=last_ids, past_key_values=past_short,
                            position_ids=pos_ids, cache_position=cache_pos,
                            output_attentions=True, use_cache=False, return_dict=True,
                        )
                finally:
                    switch_layers(loaded_impl)
                attentions = scoring.attentions
                del prefill, scoring, past_short

            drop = compute_drop(attentions, args.obs_window, args.top_k)
            drops.append(drop)
            Ts.append(T)
            rec = {"K": int(k), "id": ex_i, "T": T, "drop": float(drop)}
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
            if not args.two_pass:
                del out
            torch.cuda.empty_cache()

        mean_drop = sum(drops) / max(1, len(drops))
        var = sum((d - mean_drop) ** 2 for d in drops) / max(1, len(drops))
        std = var ** 0.5
        mean_T = sum(Ts) / max(1, len(Ts))
        summary.append((int(k), mean_T, mean_drop, std))
        print(f"K={k}: N={len(drops)}  mean_T={mean_T:.0f}  mean_drop={mean_drop:+.4f}  std={std:.4f}  elapsed={time.time()-t0:.0f}s")

    print(f"\n# K-interpolation summary\n")
    print(f"| K | N | mean T | mean drop | std drop |")
    print(f"|---:|---:|---:|---:|---:|")
    for k, mT, md, sd in summary:
        print(f"| {k} | {args.n_per_dataset} | {mT:.0f} | {md:+.4f} | {sd:.4f} |")
    out_f.close()


if __name__ == "__main__":
    main()
