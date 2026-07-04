"""Test refined a priori partition predictors.

Beyond entropy, compute on each task:

- head_agreement: mean Jaccard similarity of top-k attention positions across
  pairs of heads in the same layer, averaged over layers
- top4_mass: fraction of attention mass in top 4 keys (sharpness at very small k)
- query_consistency: how similarly do different obs_window queries pick targets
  (mean Jaccard over top-k between queries within a layer/head)

H3 prediction: capacity-bound tasks have HIGH head agreement (everyone
attends to the same needle); dilution-prone tasks have LOW head agreement
(integration across many positions, each head picks different ones).
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


def head_agreement_layer(attn_layer, top_k, obs_window):
    """attn_layer: [H, obs_window, T]. Mean Jaccard of top-k position sets across head pairs,
    aggregated over the obs_window queries."""
    H, W, T = attn_layer.shape
    # mean attention from the window per head → [H, T]
    head_mean = attn_layer.mean(dim=1)  # [H, T]
    top_idx = head_mean.topk(min(top_k, T), dim=-1).indices  # [H, top_k]
    sets = [set(top_idx[h].cpu().tolist()) for h in range(H)]
    pairs = list(itertools.combinations(range(H), 2))
    if not pairs:
        return 0.0
    return sum(jaccard(sets[i], sets[j]) for i, j in pairs) / len(pairs)


def query_consistency_layer(attn_layer, top_k):
    """Variation across queries in obs_window: how often do consecutive queries pick the same top-k?"""
    H, W, T = attn_layer.shape
    consistencies = []
    for h in range(H):
        per_query_top = attn_layer[h].topk(min(top_k, T), dim=-1).indices  # [W, top_k]
        sets = [set(per_query_top[q].cpu().tolist()) for q in range(W)]
        pairs = list(itertools.combinations(range(W), 2))
        if not pairs:
            continue
        consistencies.append(sum(jaccard(sets[i], sets[j]) for i, j in pairs) / len(pairs))
    return sum(consistencies) / max(1, len(consistencies))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="4096")
    p.add_argument("--tasks", default="niah_multikey_3,vt,fwe,qa_1,qa_2,niah_multivalue,niah_multiquery")
    p.add_argument("--n_per_task", type=int, default=20)
    p.add_argument("--obs_window", type=int, default=32)
    p.add_argument("--top_k", type=int, default=32)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--out", default="experiments/results/head_agreement.jsonl")
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

        layer_ha = []
        layer_top4 = []
        layer_qc = []
        for a in out.attentions:
            window = a[0, :, -args.obs_window:, :]  # [H, obs_window, T]
            layer_ha.append(head_agreement_layer(window, args.top_k, args.obs_window))
            top4, _ = window.topk(min(4, T), dim=-1)
            layer_top4.append(top4.sum(dim=-1).mean().item())
            layer_qc.append(query_consistency_layer(window, args.top_k))

        rec = {
            "task": ex["task"],
            "T": T,
            "head_agreement_top32": sum(layer_ha) / len(layer_ha),
            "top4_mass": sum(layer_top4) / len(layer_top4),
            "query_consistency_top32": sum(layer_qc) / len(layer_qc),
        }
        out_f.write(json.dumps(rec) + "\n")
        out_f.flush()
        del out
        torch.cuda.empty_cache()
        if (ex_i + 1) % 10 == 0:
            print(f"  [{ex_i+1}/{len(rows_to_run)}]")

    out_f.close()
    print("done")

    by_task = {}
    with open(args.out) as f:
        for line in f:
            r = json.loads(line)
            by_task.setdefault(r["task"], []).append(r)
    print("\n## Refined predictors by task")
    print("| task | n | head_agreement_top32 | top4_mass | query_consistency_top32 |")
    print("|---|---:|---:|---:|---:|")
    for task in sorted(by_task):
        rs = by_task[task]
        m_ha = sum(r["head_agreement_top32"] for r in rs) / len(rs)
        m_t4 = sum(r["top4_mass"] for r in rs) / len(rs)
        m_qc = sum(r["query_consistency_top32"] for r in rs) / len(rs)
        print(f"| {task} | {len(rs)} | {m_ha:.4f} | {m_t4:.4f} | {m_qc:.4f} |")


if __name__ == "__main__":
    main()
