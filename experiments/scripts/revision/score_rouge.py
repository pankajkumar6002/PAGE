#!/usr/bin/env python3
"""ROUGE-L F for LongBench gov_report, following LongBench metrics.py. Needs rouge==1.0.1. Usage: python -I score_rouge.py <in.jsonl> <out.jsonl>"""
import json
import sys

from rouge import Rouge

TASKS = {"gov_report"}


def rouge_score(prediction, ground_truth):
    try:
        return Rouge().get_scores([prediction], [ground_truth], avg=True)["rouge-l"]["f"], False
    except Exception:
        return 0.0, True


def main(src, dst):
    n = errors = 0
    with open(src) as f, open(dst, "w") as out:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r["task"] not in TASKS:
                continue
            best, failed = 0.0, False
            for a in r["answers"]:
                s, e = rouge_score(r["pred_plain"], a)
                best, failed = max(best, s), failed or e
            errors += failed
            n += 1
            out.write(json.dumps({"task": r["task"], "id": r["id"], "budget": r["budget"], "rougeL": best, "error": failed}) + "\n")
    print(f"scored {n} rows, rows with a rouge exception {errors}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
