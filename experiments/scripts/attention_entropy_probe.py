"""Measure attention entropy per RULER task.

H3 predicts that tasks with HIGH attention entropy (diffuse attention, dilution-prone)
will satisfy H1 (rho > threshold), while tasks with LOW entropy (sharp attention to
few positions, capacity-bound) will not.

This script computes the average attention entropy from the last `obs_window`
queries to all prompt keys, averaged over a small sample per task.
"""
import argparse
import json
import os
import time

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="4096")
    p.add_argument("--tasks", default="niah_multikey_3,vt,fwe,qa_1,qa_2,niah_multivalue,niah_multiquery")
    p.add_argument("--n_per_task", type=int, default=20)
    p.add_argument("--obs_window", type=int, default=32)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--out", default="/home/smlab/projects/eff-nn/experiments/results/entropy_probe.jsonl")
    args = p.parse_args()

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device).eval()

    ds = load_dataset("simonjegou/ruler", args.config, split="test")
    tasks = args.tasks.split(",")
    rows_to_run = []
    for task in tasks:
        sub = ds.filter(lambda r: r["task"] == task)
        sub = sub.select(range(min(args.n_per_task, len(sub))))
        for r in sub:
            rows_to_run.append(r)
    print(f"probing {len(rows_to_run)} examples across {tasks}")

    out_f = open(args.out, "w")
    for ex_i, ex in enumerate(rows_to_run):
        prompt = build_prompt(tokenizer, ex)
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        T = ids.shape[1]

        with torch.no_grad():
            out = model(input_ids=ids, output_attentions=True, use_cache=False, return_dict=True)
        # attention: tuple of L layers, each [B=1, H, T, T]
        # For each layer, take attention from last obs_window queries
        per_layer_ent = []
        per_layer_topk_mass = []
        for a in out.attentions:
            window = a[0, :, -args.obs_window:, :]  # [H, obs_window, T]
            # entropy per (head, query)
            probs = window.clamp(min=1e-12)
            ent = -(probs * probs.log()).sum(dim=-1)  # [H, obs_window]
            per_layer_ent.append(ent.mean().item())
            # top-k mass: how much mass is in top 32 keys (concentration)
            top32, _ = window.topk(min(32, T), dim=-1)
            per_layer_topk_mass.append(top32.sum(dim=-1).mean().item())

        avg_ent = sum(per_layer_ent) / len(per_layer_ent)
        avg_top32 = sum(per_layer_topk_mass) / len(per_layer_topk_mass)
        # max possible entropy at T: log(T)
        import math
        norm_ent = avg_ent / math.log(max(T, 2))

        rec = {
            "task": ex["task"],
            "T": T,
            "avg_attention_entropy": avg_ent,
            "normalized_entropy": norm_ent,
            "top32_mass": avg_top32,
        }
        out_f.write(json.dumps(rec) + "\n")
        out_f.flush()

        del out
        torch.cuda.empty_cache()
        if (ex_i + 1) % 10 == 0:
            print(f"  [{ex_i+1}/{len(rows_to_run)}]")

    out_f.close()
    print("done")

    # summary
    by_task = {}
    with open(args.out) as f:
        for line in f:
            r = json.loads(line)
            by_task.setdefault(r["task"], []).append(r)
    print("\n## Attention statistics by task")
    print("| task | n | mean normalized entropy | mean top-32 mass |")
    print("|---|---:|---:|---:|")
    for task in sorted(by_task):
        rs = by_task[task]
        m_ent = sum(r["normalized_entropy"] for r in rs) / len(rs)
        m_top32 = sum(r["top32_mass"] for r in rs) / len(rs)
        print(f"| {task} | {len(rs)} | {m_ent:.4f} | {m_top32:.4f} |")


if __name__ == "__main__":
    main()
