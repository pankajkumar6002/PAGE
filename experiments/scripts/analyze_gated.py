"""Analyze gated vs plain eviction results.

Produces per-task and pooled accuracy curves; the headline metric is the gap
between gated and plain at each budget on the mixed suite. Gated must
dominate on a mixed suite where capacity-bound tasks (NIAH-MK3) are
included; plain collapses on those at low budgets.
"""
import argparse
import json
from collections import defaultdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--tau_sweep", default="",
                   help="comma-separated tau values to evaluate post-hoc (e.g. 0.04,0.06,0.08)")
    args = p.parse_args()

    rows = [json.loads(l) for l in open(args.inp)]
    if not rows:
        print("no rows")
        return

    # per task and budget
    by_task_budget_plain = defaultdict(list)
    by_task_budget_gated = defaultdict(list)
    by_task_budget_gate_open = defaultdict(list)
    for r in rows:
        k = (r["task"], r["budget"])
        by_task_budget_plain[k].append(r["correct_plain"])
        by_task_budget_gated[k].append(r["correct_gated"])
        by_task_budget_gate_open[k].append(r["gate_open"])

    tasks = sorted({k[0] for k in by_task_budget_plain})
    budgets = sorted({k[1] for k in by_task_budget_plain}, reverse=True)

    lines = []
    lines.append(f"# Gated vs plain eviction analysis\n")
    lines.append(f"- input: {args.inp}")
    lines.append(f"- tasks: {tasks}")
    lines.append(f"- budgets: {budgets}\n")

    # per task accuracy curves
    for task in tasks:
        lines.append(f"## task = {task}")
        n = len(by_task_budget_plain[(task, budgets[0])])
        n_gate_open = sum(by_task_budget_gate_open[(task, budgets[0])]) / max(1, n)
        lines.append(f"- N = {n}; gate-open fraction = {n_gate_open:.3f}")
        lines.append(f"| budget | plain acc | gated acc | delta (gated - plain) |")
        lines.append(f"|---:|---:|---:|---:|")
        for b in budgets:
            p_acc = sum(by_task_budget_plain[(task, b)]) / max(1, len(by_task_budget_plain[(task, b)]))
            g_acc = sum(by_task_budget_gated[(task, b)]) / max(1, len(by_task_budget_gated[(task, b)]))
            lines.append(f"| {b:.4g} | {p_acc:.3f} | {g_acc:.3f} | {g_acc - p_acc:+.3f} |")
        lines.append("")

    # mixed suite (pooled across tasks)
    lines.append(f"## Mixed suite (pooled across {tasks})")
    lines.append(f"| budget | plain mean | gated mean | delta | n_gate_open / N |")
    lines.append(f"|---:|---:|---:|---:|---:|")
    for b in budgets:
        all_plain = []
        all_gated = []
        all_gate_open = []
        for task in tasks:
            all_plain.extend(by_task_budget_plain[(task, b)])
            all_gated.extend(by_task_budget_gated[(task, b)])
            all_gate_open.extend(by_task_budget_gate_open[(task, b)])
        if not all_plain:
            continue
        p_acc = sum(all_plain) / len(all_plain)
        g_acc = sum(all_gated) / len(all_gated)
        n_open = sum(all_gate_open)
        lines.append(
            f"| {b:.4g} | {p_acc:.3f} | {g_acc:.3f} | "
            f"{g_acc - p_acc:+.3f} | {n_open}/{len(all_plain)} |"
        )
    lines.append("")

    # headline (mean over budgets < 1.0)
    eviction_budgets = [b for b in budgets if b < 1.0]
    mean_plain = 0.0
    mean_gated = 0.0
    n = 0
    for b in eviction_budgets:
        for task in tasks:
            mean_plain += sum(by_task_budget_plain[(task, b)])
            mean_gated += sum(by_task_budget_gated[(task, b)])
            n += len(by_task_budget_plain[(task, b)])
    if n:
        lines.append(f"## Headline (mean over all eviction budgets, mixed suite)")
        lines.append(f"- plain mean accuracy: {mean_plain/n:.3f}")
        lines.append(f"- gated mean accuracy: {mean_gated/n:.3f}")
        lines.append(f"- **delta (gated - plain): {(mean_gated - mean_plain)/n:+.3f}**")
        threshold_passed = (mean_gated - mean_plain) / n >= 0.03
        lines.append(f"- Week-4 hard checkpoint (≥+3pp): {'PASSED' if threshold_passed else 'FAILED'}")
    lines.append("")

    out = "\n".join(lines)
    print(out)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
