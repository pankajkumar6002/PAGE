#!/usr/bin/env python
"""B2: gating runs beyond RULER. Pre-registered in preregistration/revision/b2_beyond_ruler_prereg.md.

Datasets:
  quality        emozilla/quality validation, 300 questions from a seed-0 shuffle
  lbv2           zai-org/LongBench-v2, contexts up to --max_context_tokens
  longbench      Xnhyacinth/LongBench subtasks given by --tasks

Only the plain SnapKV arm is decoded at each budget. The gated arm follows from the
gate identity. Per-layer agreement is logged so early-layer agreement gets a held-out
test. Reuses longbench_gating.py's prefill, masks and decoding unchanged.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time

import torch

SRC = os.environ.get("PAGE_SRC", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SRC)
import longbench_gating as lg  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

LETTERS = "ABCD"
MCQ_TAIL = "Answer with the letter of the correct option only."
# B2b, preregistration/revision/b2b_reasoning_prompt_prereg.md
MCQ_TAIL_REASONING = ("First quote the sentence from the text that best supports the answer. Then give the letter "
                      "of the correct option on a final line in the form Answer: X. Keep it under 60 words.")
TAIL = MCQ_TAIL


def mcq_row(context, question, options, gold_letter, doc, task, extra=None):
    opts = "\n".join(f"{LETTERS[i]}) {o}" for i, o in enumerate(options))
    r = {"context": context, "question": f"{question}\n{opts}", "answer_prefix": TAIL, "_task": task, "_doc": doc, "answers": [gold_letter], "_mcq": True}
    if extra:
        r.update(extra)
    return r


def load_rows(args):
    rows = []
    if args.dataset == "quality":
        ds = load_dataset("emozilla/quality", split="validation").shuffle(seed=0)
        for r in ds.select(range(min(args.n, len(ds)))):
            doc = hashlib.sha1(r["article"].encode()).hexdigest()[:12]
            rows.append(mcq_row(r["article"], r["question"], r["options"], LETTERS[int(r["answer"])], doc, "quality", {"_hard": bool(r["hard"])}))
    elif args.dataset == "lbv2":
        ds = load_dataset("zai-org/LongBench-v2", split="train")
        print("lbv2 fields:", ds.column_names, flush=True)
        for r in ds:
            doc = hashlib.sha1(r["context"].encode()).hexdigest()[:12]
            rows.append(mcq_row(r["context"], r["question"], [r["choice_A"], r["choice_B"], r["choice_C"], r["choice_D"]], r["answer"], doc, "lbv2", {"_domain": r.get("domain")}))
    else:
        for task in args.tasks.split(","):
            ds = load_dataset("Xnhyacinth/LongBench", task, split="test")
            for r in ds.select(range(min(args.n, len(ds)))):
                r = dict(r)
                r["_task"], r["_doc"], r["_mcq"] = task, None, False
                rows.append(r)
    return rows


def parse_letter(pred, style="letter"):
    if style == "reasoning":
        found = re.findall(r"Answer:\s*\(?([A-D])\)?", pred)
        return found[-1] if found else None
    m = re.search(r"\b([A-D])\b", pred)
    return m.group(1) if m else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["quality", "lbv2", "longbench"])
    p.add_argument("--tasks", default="")
    p.add_argument("--model", required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--budgets", default="1.0,0.5,0.25,0.125,0.0625")
    p.add_argument("--max_new", type=int, default=128)
    p.add_argument("--mcq_max_new", type=int, default=8)
    p.add_argument("--attn_impl", default="eager", choices=["eager", "sdpa"])
    p.add_argument("--two_pass", action="store_true")
    p.add_argument("--max_context_tokens", type=int, default=32000)
    p.add_argument("--obs_window", type=int, default=32)
    p.add_argument("--n_sink", type=int, default=4)
    p.add_argument("--top_k", type=int, default=32)
    p.add_argument("--validate_log", default=None, help="check drop, T and predictions against this LongBench log, then exit")
    p.add_argument("--validate_n", type=int, default=5)
    p.add_argument("--mcq_style", default="letter", choices=["letter", "reasoning"])
    args = p.parse_args()

    global TAIL
    if args.mcq_style == "reasoning":
        TAIL = MCQ_TAIL_REASONING
        args.mcq_max_new = 128
    budgets = [float(b) for b in args.budgets.split(",")]
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation=args.attn_impl).to(device).eval()
    rows = load_rows(args)

    logged = {}
    if args.validate_log:
        rows = rows[: args.validate_n]
        with open(args.validate_log) as f:
            for line in f:
                r = json.loads(line)
                logged[(r["task"], r["id"], float(r["budget"]))] = r

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fout = open(args.out, "a")
    t0 = time.time()
    n_skip, n_oom = 0, 0
    per_task_i = {}
    fails = 0
    for ex_i, ex in enumerate(rows):
        task = ex["_task"]
        ti = per_task_i.get(task, 0)
        per_task_i[task] = ti + 1
        prompt = lg.build_prompt(tok, ex)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False, truncation=False).input_ids.to(device)
        T = ids.shape[1]
        if T > args.max_context_tokens:
            n_skip += 1
            continue
        try:
            keys, values, score, drop, per_layer = lg.prefill_and_score_with_agreement(model, ids, args.obs_window, args.top_k, device, two_pass=args.two_pass)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            n_oom += 1
            continue
        if ex.get("_mcq"):
            max_new = args.mcq_max_new
        else:
            max_new = max(int(ex.get("max_new_tokens") or args.max_new), args.max_new)
        for b in budgets:
            pred, n_kept = lg.evaluate_one(model, tok, ids, keys, values, score, T, b, max_new, device, args.obs_window, args.n_sink, do_evict=True)
            if ex.get("_mcq"):
                letter = parse_letter(pred, args.mcq_style)
                ok, parse_fail = letter == ex["answers"][0], letter is None
            else:
                ok, parse_fail = lg.is_correct(pred, ex["answers"]), False
            rec = {"id": ex_i, "task_index": ti, "task": task, "doc": ex.get("_doc"), "budget": b, "T": T, "drop": float(drop), "n_kept_plain": n_kept,
                   "answers": ex["answers"], "pred_plain": pred[:4000], "correct_plain": bool(ok), "parse_fail": bool(parse_fail)}
            if "_hard" in ex:
                rec["hard"] = ex["_hard"]
            if "_domain" in ex:
                rec["domain"] = ex["_domain"]
            if b == 1.0:
                rec["per_layer"] = per_layer
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            if args.validate_log:
                old = logged.get((task, ex_i, b))
                if old is not None:
                    same = old["pred_plain"][:200] == pred[:200]
                    dd = abs(old["drop"] - drop)
                    print(f"validate {task}/{ex_i} b={b}: T {old['T']} vs {T}, |dD|={dd:.5f}, pred_match={same}", flush=True)
                    fails += int(old["T"] != T or dd > 0.005 or (b in (1.0, 0.25) and not same))
        del keys, values, score
        torch.cuda.empty_cache()
        if (ex_i + 1) % 10 == 0:
            print(f"[{ex_i + 1}/{len(rows)}] {task} T={T} {time.time() - t0:.0f}s skip={n_skip} oom={n_oom}", flush=True)
    fout.close()
    print(f"b2 done rows={len(rows)} skipped={n_skip} oom={n_oom} validation_mismatches={fails}", flush=True)


if __name__ == "__main__":
    main()
