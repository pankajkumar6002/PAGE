"""Per-layer head agreement: is there a specific layer band where
dilution-prone vs capacity-bound tasks separate?

For each example, report mean Jaccard top-k agreement across heads PER LAYER,
then aggregate to find the layer-bin (early / middle / late) with the
strongest task-type separation.
"""
import argparse
import itertools
import json

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    H, W, T = attn_layer.shape
    head_mean = attn_layer.mean(dim=1)
    top_idx = head_mean.topk(min(top_k, T), dim=-1).indices
    sets = [set(top_idx[h].cpu().tolist()) for h in range(H)]
    pairs = list(itertools.combinations(range(H), 2))
    if not pairs:
        return 0.0
    return sum(jaccard(sets[i], sets[j]) for i, j in pairs) / len(pairs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="4096")
    p.add_argument("--tasks", default="niah_multikey_3,vt,fwe,qa_1,qa_2,niah_multivalue,niah_multiquery")
    p.add_argument("--n_per_task", type=int, default=20)
    p.add_argument("--obs_window", type=int, default=32)
    p.add_argument("--top_k", type=int, default=32)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--out", default="experiments/results/per_layer_agreement.jsonl")
    p.add_argument("--two_pass", action="store_true",
                   help="Use SDPA pass-1 + eager pass-2 to avoid OOM at long contexts.")
    args = p.parse_args()

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    loaded_impl = "sdpa" if args.two_pass else "eager"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation=loaded_impl,
        trust_remote_code=True,
    ).to(device).eval()

    def switch_layers(impl):
        for layer in model.model.layers:
            for attr_name in ("self_attn", "attention"):
                attn_mod = getattr(layer, attr_name, None)
                if attn_mod is not None:
                    try:
                        attn_mod.config._attn_implementation = impl
                    except AttributeError:
                        pass

    from transformers.cache_utils import DynamicCache

    ds = load_dataset("simonjegou/ruler", args.config, split="test")
    tasks = args.tasks.split(",")
    rows_to_run = []
    for task in tasks:
        sub = ds.filter(lambda r: r["task"] == task)
        sub = sub.select(range(min(args.n_per_task, len(sub))))
        for r in sub:
            rows_to_run.append(r)
    print(f"probing {len(rows_to_run)} examples")

    out_f = open(args.out, "w")
    for ex_i, ex in enumerate(rows_to_run):
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
                k, v = layer.keys, layer.values
                past_short.update(
                    k[:, :, : T - args.obs_window, :].clone(),
                    v[:, :, : T - args.obs_window, :].clone(),
                    i,
                )
            last_ids = ids[:, -args.obs_window:]
            pos_ids = torch.arange(T - args.obs_window, T, device=device).unsqueeze(0)
            cache_pos = torch.arange(T - args.obs_window, T, device=device)
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
            attentions = scoring.attentions
            del prefill, scoring, past_short
            torch.cuda.empty_cache()
        per_layer = []
        for a in attentions:
            if a.shape[2] == args.obs_window:
                window = a[0]
            else:
                window = a[0, :, -args.obs_window:, :]
            per_layer.append(head_agreement_layer(window, args.top_k))
        rec = {
            "task": ex["task"],
            "T": T,
            "head_agreement_per_layer": per_layer,
        }
        out_f.write(json.dumps(rec) + "\n")
        out_f.flush()
        if not args.two_pass:
            del out
        torch.cuda.empty_cache()
        if (ex_i + 1) % 10 == 0:
            print(f"  [{ex_i+1}/{len(rows_to_run)}]")

    out_f.close()
    print("done")

    # Aggregate by task and layer bin
    from collections import defaultdict
    by_task_layer = defaultdict(list)
    L = None
    with open(args.out) as f:
        for line in f:
            r = json.loads(line)
            pl = r["head_agreement_per_layer"]
            L = len(pl)
            for i, v in enumerate(pl):
                by_task_layer[(r["task"], i)].append(v)

    early = list(range(0, L // 3))
    middle = list(range(L // 3, 2 * L // 3))
    late = list(range(2 * L // 3, L))

    tasks_sorted = sorted({k[0] for k in by_task_layer})
    print(f"\n## Per-layer-bin head agreement (L={L} layers)")
    print("| task | early | middle | late |")
    print("|---|---:|---:|---:|")
    for task in tasks_sorted:
        def avg(layers):
            vals = []
            for i in layers:
                vals.extend(by_task_layer[(task, i)])
            return sum(vals) / max(1, len(vals))
        e, m, lt = avg(early), avg(middle), avg(late)
        print(f"| {task} | {e:.4f} | {m:.4f} | {lt:.4f} |")


if __name__ == "__main__":
    main()
