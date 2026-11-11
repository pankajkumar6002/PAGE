"""Analyze the near-tie-distractor-count ablation.

Inputs:
  - drops_niah_mk_ablation.jsonl  (per-input head_agreement_per_layer)
  - gated_4k_niah_mk_ablation.jsonl (per-input per-budget correct_plain / correct_gated, drop, gate_open)

Outputs (printed and written to summary md):
  Per task:
    - mean D (early - late, third bins)
    - rho_plain  = fraction of inputs where ANY b<1.0 plain-eviction is correct AND b=1.0 is wrong
    - rho_gated  = same but for gated
    - gate-open fraction (from any-budget row; gate_open is per-input, not per-budget)
"""
import argparse
import json
from collections import defaultdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--drops", default="experiments/results/drops_niah_mk_ablation.jsonl")
    p.add_argument("--gated", default="experiments/results/gated_4k_niah_mk_ablation.jsonl")
    p.add_argument("--out", default="experiments/results/niah_mk_ablation.md")
    args = p.parse_args()

    # Compute mean D per task from drops file
    per_task_drops = defaultdict(list)
    for line in open(args.drops):
        r = json.loads(line)
        pl = r["head_agreement_per_layer"]
        L = len(pl)
        third = max(1, L // 3)
        early = sum(pl[:third]) / third
        late = sum(pl[L - third:]) / third
        per_task_drops[r["task"]].append(early - late)

    mean_D = {t: sum(arr) / len(arr) for t, arr in per_task_drops.items()}

    # Load gated rows
    rows = [json.loads(l) for l in open(args.gated)]
    if not rows:
        print("no gated rows")
        return

    # Group by (task, id) -> {budget: (correct_plain, correct_gated, drop, gate_open)}
    by_task_id = defaultdict(dict)
    for r in rows:
        by_task_id[(r["task"], r["id"])][r["budget"]] = (
            bool(r["correct_plain"]),
            bool(r["correct_gated"]),
            float(r["drop"]),
            bool(r["gate_open"]),
        )

    tasks = sorted({t for (t, _) in by_task_id})
    budgets_all = sorted({b for d in by_task_id.values() for b in d.keys()}, reverse=True)
    full_b = max(budgets_all)
    lower_budgets = [b for b in budgets_all if b < full_b]

    lines = []
    lines.append("# Near-tie distractor ablation: niah_multikey_{1,2,3}")
    lines.append("")
    lines.append("Qwen2.5-1.5B-Instruct, RULER 4K, N=50 per task. tau=0.07. Budgets: " + ", ".join(f"{b:g}" for b in budgets_all))
    lines.append("")

    # Per-task summary table
    lines.append("## Per-task mean D and recovery rates")
    lines.append("")
    lines.append("| task | N | mean D | rho_plain | rho_gated | gate-open frac |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    per_task_summary = {}
    for task in tasks:
        ids = [iid for (t, iid) in by_task_id if t == task]
        N = len(ids)
        # rho_plain: id is recovered if any b < full is correct_plain and full is wrong (plain)
        rho_plain_count = 0
        rho_gated_count = 0
        gate_open_count = 0
        for iid in ids:
            d = by_task_id[(task, iid)]
            if full_b not in d:
                continue
            full_plain_correct, full_gated_correct, drop, gate_open = d[full_b]
            gate_open_count += int(gate_open)
            # plain at full == always full KV (plain treats b=1.0 as full)
            if not full_plain_correct:
                for b in lower_budgets:
                    if b in d and d[b][0]:
                        rho_plain_count += 1
                        break
            if not full_gated_correct:
                for b in lower_budgets:
                    if b in d and d[b][1]:
                        rho_gated_count += 1
                        break
        rho_plain = rho_plain_count / N if N else 0.0
        rho_gated = rho_gated_count / N if N else 0.0
        gate_open_frac = gate_open_count / N if N else 0.0
        D = mean_D.get(task, float("nan"))
        per_task_summary[task] = (N, D, rho_plain, rho_gated, gate_open_frac)
        lines.append(f"| {task} | {N} | {D:+.4f} | {rho_plain:.3f} | {rho_gated:.3f} | {gate_open_frac:.3f} |")

    lines.append("")

    # Per-task per-budget accuracy curves (plain vs gated)
    lines.append("## Per-task per-budget accuracy")
    lines.append("")
    for task in tasks:
        ids = [iid for (t, iid) in by_task_id if t == task]
        lines.append(f"### {task}  (N={len(ids)})")
        lines.append("")
        lines.append("| budget | plain acc | gated acc | delta |")
        lines.append("|---:|---:|---:|---:|")
        for b in budgets_all:
            plain_correct = [by_task_id[(task, iid)][b][0] for iid in ids if b in by_task_id[(task, iid)]]
            gated_correct = [by_task_id[(task, iid)][b][1] for iid in ids if b in by_task_id[(task, iid)]]
            if not plain_correct:
                continue
            p_acc = sum(plain_correct) / len(plain_correct)
            g_acc = sum(gated_correct) / len(gated_correct)
            lines.append(f"| {b:g} | {p_acc:.3f} | {g_acc:.3f} | {g_acc - p_acc:+.3f} |")
        lines.append("")

    # Per-task drop summary
    lines.append("## Per-task D (head-agreement drop) detail")
    lines.append("")
    lines.append("| task | mean D | min D | median D | max D | n |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for task in tasks:
        arr = sorted(per_task_drops[task])
        if not arr:
            continue
        n = len(arr)
        med = arr[n // 2]
        lines.append(f"| {task} | {sum(arr)/n:+.4f} | {arr[0]:+.4f} | {med:+.4f} | {arr[-1]:+.4f} | {n} |")
    lines.append("")

    # Gradient verdict
    lines.append("## Gradient verdict")
    lines.append("")
    D_vals = [(t, per_task_summary[t][1]) for t in tasks]
    rho_vals = [(t, per_task_summary[t][2]) for t in tasks]
    gate_vals = [(t, per_task_summary[t][4]) for t in tasks]
    lines.append(f"- D order: " + " > ".join(f"{t}({d:+.4f})" for t, d in sorted(D_vals, key=lambda x: -x[1])))
    lines.append(f"- rho_plain order: " + " > ".join(f"{t}({r:.3f})" for t, r in sorted(rho_vals, key=lambda x: -x[1])))
    lines.append(f"- gate-open frac order: " + " > ".join(f"{t}({g:.3f})" for t, g in sorted(gate_vals, key=lambda x: -x[1])))
    lines.append("")
    expected = ["niah_multikey_1", "niah_multikey_2", "niah_multikey_3"]
    if all(t in per_task_summary for t in expected):
        D_order_correct = per_task_summary[expected[0]][1] > per_task_summary[expected[1]][1] > per_task_summary[expected[2]][1]
        rho_order_correct = per_task_summary[expected[0]][2] >= per_task_summary[expected[1]][2] >= per_task_summary[expected[2]][2]
        gate_order_correct = per_task_summary[expected[0]][4] >= per_task_summary[expected[1]][4] >= per_task_summary[expected[2]][4]
        lines.append(f"- Predicted gradient: niah_mk_1 > niah_mk_2 > niah_mk_3 on D, rho_plain, gate-open frac.")
        lines.append(f"  - D gradient holds: {D_order_correct}")
        lines.append(f"  - rho_plain gradient holds (weakly): {rho_order_correct}")
        lines.append(f"  - gate-open gradient holds (weakly): {gate_order_correct}")

    out_text = "\n".join(lines) + "\n"
    with open(args.out, "w") as f:
        f.write(out_text)
    print(out_text)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
