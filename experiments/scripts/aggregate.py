"""Aggregate all sweep results into one summary table.

Walks experiments/results/ for *.jsonl matching known sweep file patterns,
computes per-(file, budget) accuracy and rho, prints a single comparison table.
"""
import argparse
import glob
import json
import os
from collections import defaultdict


def load(path):
    by_id = defaultdict(dict)
    Ts = []
    tasks = set()
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "correct" not in r or "budget" not in r:
                return None, None, None
            by_id[r["id"]][r["budget"]] = r["correct"]
            Ts.append(r["T"])
            if "task" in r:
                tasks.add(r["task"])
    return by_id, Ts, tasks


def summarize(by_id):
    N = len(by_id)
    if not N:
        return {}, 0.0, 0
    budgets = sorted({b for v in by_id.values() for b in v.keys()}, reverse=True)
    acc = {b: sum(v.get(b, False) for v in by_id.values()) / N for b in budgets}
    full_b = max(budgets)
    n_full_wrong = sum(1 for v in by_id.values() if not v.get(full_b, False))
    n_h1 = sum(1 for v in by_id.values()
               if not v.get(full_b, False) and any(v.get(b, False) for b in budgets if b < full_b))
    rho = n_h1 / N if N else 0.0
    best_low_b = max((b for b in budgets if b < full_b), default=None, key=lambda b: acc.get(b, -1))
    return acc, rho, n_full_wrong, n_h1, best_low_b, N


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="/home/smlab/projects/eff-nn/experiments/results")
    p.add_argument("--out", default="/home/smlab/projects/eff-nn/experiments/results/aggregate.md")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.jsonl")))
    rows = []
    for fp in files:
        if os.path.getsize(fp) == 0:
            continue
        name = os.path.basename(fp).replace(".jsonl", "")
        by_id, Ts, tasks = load(fp)
        if by_id is None or not by_id:
            continue
        acc, rho, nfw, nh1, best_low, N = summarize(by_id)
        rows.append({
            "name": name,
            "N": N,
            "T_mean": sum(Ts) / max(1, len(Ts)),
            "tasks": ",".join(sorted(tasks)) if tasks else "",
            "acc_full": acc.get(max(acc.keys()), float("nan")),
            "best_low_b": best_low,
            "acc_best_low": acc.get(best_low, float("nan")) if best_low else float("nan"),
            "rho": rho,
            "n_full_wrong": nfw,
            "n_recovered": nh1,
        })

    lines = []
    lines.append("# Aggregate H1 results\n")
    lines.append("| run | N | T̄ | task | acc@full | best b<1 (acc) | rho | wrong@full | recovered |")
    lines.append("|---|---:|---:|---|---:|---:|---:|---:|---:|")
    for r in rows:
        bl = f"{r['best_low_b']:.4g} ({r['acc_best_low']:.3f})" if r["best_low_b"] is not None else "—"
        lines.append(
            f"| {r['name']} | {r['N']} | {r['T_mean']:.0f} | {r['tasks'] or '—'} | "
            f"{r['acc_full']:.3f} | {bl} | {r['rho']:.4f} | {r['n_full_wrong']} | {r['n_recovered']} |"
        )
    out = "\n".join(lines)
    print(out)
    with open(args.out, "w") as f:
        f.write(out + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
