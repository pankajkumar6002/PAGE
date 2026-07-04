"""Compute H1 statistics from the sweep results.

Reads experiments/results/h1_sweep.jsonl with rows
  {id, budget, T, n_kept, gold, pred, correct}
and produces:
  - per-budget accuracy
  - rho_KV = fraction of ids where SOME budget < 1.0 is correct AND budget == 1.0 is wrong
  - rho_KV_strict = same as rho_KV but using majority/best-of-budgets vs full
  - per-budget marginal: for each budget b < 1.0, fraction of ids where b correct but 1.0 wrong
  - flip table: for each pair (b1, b2) the +/- diff
"""
import argparse
import json
from collections import defaultdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", default="experiments/results/h1_sweep.jsonl")
    p.add_argument("--out", default="experiments/results/h1_report.md")
    args = p.parse_args()

    by_id = defaultdict(dict)  # id -> {budget -> bool}
    Ts = {}
    n_kept = defaultdict(dict)
    preds = defaultdict(dict)
    golds = {}
    with open(args.inp) as f:
        for line in f:
            r = json.loads(line)
            by_id[r["id"]][r["budget"]] = r["correct"]
            Ts[r["id"]] = r["T"]
            n_kept[r["id"]][r["budget"]] = r["n_kept"]
            preds[r["id"]][r["budget"]] = r["pred"]
            golds[r["id"]] = r["gold"]

    budgets = sorted({b for d in by_id.values() for b in d.keys()}, reverse=True)
    N = len(by_id)

    per_budget_acc = {b: sum(d.get(b, False) for d in by_id.values()) / N for b in budgets}

    full_b = 1.0
    if full_b not in budgets:
        full_b = max(budgets)

    n_full_wrong = sum(1 for d in by_id.values() if not d.get(full_b, False))
    n_any_lower_correct_when_full_wrong = sum(
        1 for d in by_id.values()
        if not d.get(full_b, False) and any(d.get(b, False) for b in budgets if b < full_b)
    )
    rho_KV = n_any_lower_correct_when_full_wrong / N if N else 0.0

    per_budget_marginal_when_full_wrong = {}
    for b in budgets:
        if b >= full_b:
            continue
        gain = sum(1 for d in by_id.values() if not d.get(full_b, False) and d.get(b, False))
        per_budget_marginal_when_full_wrong[b] = gain / N

    n_full_right_lower_wrong = {}
    for b in budgets:
        if b >= full_b:
            continue
        loss = sum(1 for d in by_id.values() if d.get(full_b, False) and not d.get(b, False))
        n_full_right_lower_wrong[b] = loss / N

    lines = []
    lines.append(f"# H1 spot-check report\n")
    lines.append(f"- N examples: **{N}**")
    lines.append(f"- Mean prompt tokens T: **{sum(Ts.values())/max(1,len(Ts)):.0f}**")
    lines.append("")
    lines.append("## Per-budget accuracy")
    lines.append("| budget | mean n_kept | accuracy |")
    lines.append("|---:|---:|---:|")
    for b in budgets:
        mean_kept = sum(n_kept[i].get(b, 0) for i in by_id) / max(1, N)
        lines.append(f"| {b:.4g} | {mean_kept:.0f} | {per_budget_acc[b]:.3f} |")
    lines.append("")
    lines.append("## H1 metric (interior accuracy maximum)")
    lines.append(f"- Inputs where full-KV (b={full_b}) is **wrong**: {n_full_wrong}/{N} = {n_full_wrong/N:.3f}")
    lines.append(f"- Of those, inputs where SOME budget < full-KV is **correct**: {n_any_lower_correct_when_full_wrong}/{N}")
    lines.append(f"- **rho_KV = {rho_KV:.4f}** (fraction of all inputs where less compute beats full-KV)")
    lines.append("")
    lines.append("## Per-budget marginal: \"correct at b, wrong at full-KV\"")
    lines.append("| budget | gain / N | loss / N | net |")
    lines.append("|---:|---:|---:|---:|")
    for b in budgets:
        if b >= full_b:
            continue
        g = per_budget_marginal_when_full_wrong[b]
        l = n_full_right_lower_wrong[b]
        lines.append(f"| {b:.4g} | +{g:.3f} | -{l:.3f} | {g-l:+.3f} |")
    lines.append("")
    lines.append("## Interpretation")
    if rho_KV >= 0.05:
        lines.append(f"- rho_KV = {rho_KV:.3f} >= 0.05 — H1 is empirically supported on this slice. Proceed.")
    elif rho_KV >= 0.02:
        lines.append(f"- rho_KV = {rho_KV:.3f} (between 0.02 and 0.05) — weak signal. Needs more examples or a different task before deciding.")
    else:
        lines.append(f"- rho_KV = {rho_KV:.3f} < 0.02 — H1 NOT supported on this slice. Investigate task choice / model / eviction policy before continuing.")
    lines.append("")

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
