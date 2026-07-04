"""Re-analyze RULER results using task-appropriate metrics.

- NIAH variants: any gold in pred (existing)
- VT (variable tracking): ALL golds in pred
- FWE (frequent words extraction): ALL golds in pred
- CWE (common words extraction): ALL golds in pred
- QA: any gold in pred

Outputs per-task accuracy by budget.
"""
import argparse
import json
import os
from collections import defaultdict


def is_correct(pred, golds, task):
    pred_l = pred.lower()
    if task in ("vt", "fwe", "cwe", "niah_multivalue"):
        return all(g.strip().lower() in pred_l for g in golds)
    return any(g.strip().lower() in pred_l for g in golds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    rows = [json.loads(l) for l in open(args.inp)]
    by_task_budget = defaultdict(list)
    per_id_per_task = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        correct = is_correct(r["pred"], r["gold"], r["task"])
        by_task_budget[(r["task"], r["budget"])].append(correct)
        per_id_per_task[r["task"]][r["id"]][r["budget"]] = correct

    tasks = sorted({k[0] for k in by_task_budget.keys()})
    budgets = sorted({k[1] for k in by_task_budget.keys()}, reverse=True)

    lines = [f"# RULER reanalysis: {os.path.basename(args.inp)}\n"]
    for task in tasks:
        N = len(per_id_per_task[task])
        lines.append(f"\n## task = {task}  (N={N})\n")
        lines.append("| budget | acc | gain@b vs full | loss@b vs full |")
        lines.append("|---:|---:|---:|---:|")
        full_b = max(budgets)
        d = per_id_per_task[task]
        for b in budgets:
            acc = sum(d[i].get(b, False) for i in d) / max(1, N)
            gain = sum(1 for i in d if not d[i].get(full_b, False) and d[i].get(b, False)) / max(1, N)
            loss = sum(1 for i in d if d[i].get(full_b, False) and not d[i].get(b, False)) / max(1, N)
            lines.append(f"| {b:.4g} | {acc:.3f} | +{gain:.3f} | -{loss:.3f} |")
        # H1 rho
        n_full_wrong = sum(1 for i in d if not d[i].get(full_b, False))
        n_h1 = sum(1 for i in d if not d[i].get(full_b, False)
                   and any(d[i].get(b, False) for b in budgets if b < full_b))
        lines.append(f"\n- rho_KV = **{n_h1/max(1,N):.4f}** ({n_h1}/{N}); wrong@full = {n_full_wrong}\n")
    out = "\n".join(lines)
    print(out)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
