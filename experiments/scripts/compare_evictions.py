"""Compare two eviction sweeps (e.g. SnapKV vs random) on the same task.

Produces a side-by-side accuracy-vs-budget table and a paired rho comparison.
"""
import argparse
import json
from collections import defaultdict


def load(path):
    by_id = defaultdict(dict)
    Ts = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            by_id[r["id"]][r["budget"]] = r["correct"]
            Ts[r["id"]] = r["T"]
    return by_id, Ts


def per_budget_acc(d):
    N = len(d)
    budgets = sorted({b for v in d.values() for b in v.keys()}, reverse=True)
    return {b: sum(v.get(b, False) for v in d.values()) / N for b in budgets}, budgets, N


def rho(d, full_b=1.0):
    N = len(d)
    if not N:
        return 0.0, 0, 0
    budgets = sorted({b for v in d.values() for b in v.keys()}, reverse=True)
    n_full_wrong = sum(1 for v in d.values() if not v.get(full_b, False))
    n_h1 = sum(1 for v in d.values()
               if not v.get(full_b, False) and any(v.get(b, False) for b in budgets if b < full_b))
    return n_h1 / N, n_full_wrong, n_h1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="path to first sweep jsonl")
    p.add_argument("--b", required=True, help="path to second sweep jsonl")
    p.add_argument("--label_a", default="A")
    p.add_argument("--label_b", default="B")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    da, _ = load(args.a)
    db, _ = load(args.b)
    acc_a, budgets_a, Na = per_budget_acc(da)
    acc_b, budgets_b, Nb = per_budget_acc(db)
    rho_a, nfw_a, nh1_a = rho(da)
    rho_b, nfw_b, nh1_b = rho(db)

    budgets = sorted(set(budgets_a) | set(budgets_b), reverse=True)

    lines = []
    lines.append(f"# Eviction comparison: {args.label_a} vs {args.label_b}\n")
    lines.append(f"- N_{args.label_a}: {Na}")
    lines.append(f"- N_{args.label_b}: {Nb}\n")
    lines.append(f"## Accuracy by budget")
    lines.append(f"| budget | {args.label_a} acc | {args.label_b} acc | delta |")
    lines.append("|---:|---:|---:|---:|")
    for bu in budgets:
        aa = acc_a.get(bu, float("nan"))
        bb = acc_b.get(bu, float("nan"))
        d = aa - bb if (aa == aa and bb == bb) else float("nan")
        lines.append(f"| {bu:.4g} | {aa:.3f} | {bb:.3f} | {d:+.3f} |")
    lines.append("")
    lines.append(f"## H1 (rho) summary")
    lines.append(f"| run | rho | full-wrong | H1 inputs |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| {args.label_a} | {rho_a:.4f} | {nfw_a} | {nh1_a} |")
    lines.append(f"| {args.label_b} | {rho_b:.4f} | {nfw_b} | {nh1_b} |")
    lines.append("")

    out_txt = "\n".join(lines)
    print(out_txt)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out_txt + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
