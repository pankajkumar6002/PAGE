"""P3-3: dump per-head-pair Jaccards so head-pair subsampling can be evaluated.

page-kv's head_agreement_probe.py averages over head pairs inside
head_agreement_layer() before writing, so the released logs cannot answer
"how many head pairs does D actually need?". This probe keeps the per-pair
values, which makes the subsampling question answerable offline afterwards
exactly as the layer question already is.

Cost note: D is O(L H^2 k), and the H^2 term is the head pairs. If a small
random subset of pairs preserves the ordering the gate depends on, the gate
gets cheaper on exactly the wide-head models where it is most expensive,
which is the deployment objection in Section limitations.

Prefill only, no decoding.
"""
import argparse
import itertools
import json
import os
import random

import torch


def build_prompt(tok, ex):
    user = ex["context"] + "\n\n" + ex["question"]
    if ex.get("answer_prefix"):
        user = user + "\n" + ex["answer_prefix"]
    msgs = [{"role": "system", "content": "Answer the question concisely."},
            {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def jaccard(a, b):
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def pair_jaccards(attn_layer, top_k):
    """attn_layer [H, W, T] -> {(i,j): jaccard} over all head pairs."""
    H, W, T = attn_layer.shape
    top = attn_layer.mean(dim=1).topk(min(top_k, T), dim=-1).indices
    sets = [set(top[h].tolist()) for h in range(H)]
    return {(i, j): jaccard(sets[i], sets[j])
            for i, j in itertools.combinations(range(H), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--config", default="4096")
    ap.add_argument("--tasks", default="niah_multikey_3,vt,fwe,qa_1")
    ap.add_argument("--n_per_task", type=int, default=30)
    ap.add_argument("--obs_window", type=int, default=32)
    ap.add_argument("--top_k", type=int, default=32)
    ap.add_argument("--device_map", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tasks = a.tasks.split(",")
    ds = load_dataset("simonjegou/ruler", a.config, split="test")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation="eager",
        device_map=a.device_map or {"": 0})
    model.eval()
    dev = next(model.parameters()).device

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    n_written = 0
    with open(a.out, "w") as fh:
        for task in tasks:
            sub = ds.filter(lambda r: r["task"] == task).select(range(a.n_per_task))
            for i, ex in enumerate(sub):
                ids = tok(build_prompt(tok, ex), return_tensors="pt").input_ids.to(dev)
                with torch.no_grad():
                    out = model(input_ids=ids, use_cache=False,
                                output_attentions=True)
                L = len(out.attentions)
                # per layer, keep the per-pair Jaccards over the observation window
                per_layer = []
                for lyr in out.attentions:
                    A = lyr[0][:, -a.obs_window:, :].float()      # [H, W, T]
                    pj = pair_jaccards(A, a.top_k)
                    per_layer.append({f"{i2}-{j2}": round(v, 6)
                                      for (i2, j2), v in pj.items()})
                fh.write(json.dumps({
                    "task": task, "id": i, "T": int(ids.shape[1]),
                    "L": L, "H": model.config.num_attention_heads,
                    "pair_jaccard_per_layer": per_layer,
                }) + "\n")
                n_written += 1
                del out
                torch.cuda.empty_cache()
            print(f"  {task}: {a.n_per_task} inputs", flush=True)
    print(f"wrote {n_written} rows -> {a.out}")


if __name__ == "__main__":
    main()
