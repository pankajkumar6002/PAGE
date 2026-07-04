"""Analyze longbench_gating.py output for realistic-workload capacity-bound test.

For each task jsonl, computes:
  - A_full (full-KV accuracy, substring metric; exact-int also for any code/count)
  - accuracy vs budget (plain and gated)
  - full-KV-correct retention at each budget (the catastrophe signal)
  - rho = Pr_x[full-KV wrong AND some b<1.0 correct] (dilution signal)
  - mean head-agreement drop D, gate-open fraction at tau
  - empirical class + gate prediction + correct/incorrect

Usage: python analyze_realistic_workload.py f1.jsonl f2.jsonl ...
Writes experiments/results/realistic_workload_partition.md
"""
import json
import sys
import statistics as st
from collections import defaultdict

TAU = 0.07
OUT_MD = "/home/smlab/projects/eff-nn/experiments/results/realistic_workload_partition.md"


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def analyze(path):
    rows = load(path)
    task = rows[0]["task"]
    by_id = defaultdict(dict)
    for r in rows:
        by_id[r["id"]][r["budget"]] = r
    ids = sorted(by_id)
    budgets = sorted({r["budget"] for r in rows}, reverse=True)
    N = len(ids)

    def correct(r):
        return bool(r["correct_plain"]), bool(r["correct_gated"])

    # accuracy vs budget
    acc_plain = {}
    acc_gated = {}
    retain = {}  # of full-correct inputs, how many still correct (plain) at budget b
    full_correct_ids = [i for i in ids if 1.0 in by_id[i] and by_id[i][1.0]["correct_plain"]]
    n_full = len(full_correct_ids)
    for b in budgets:
        cp = cg = n = 0
        for i in ids:
            r = by_id[i].get(b)
            if r is None:
                continue
            n += 1
            cp += bool(r["correct_plain"])
            cg += bool(r["correct_gated"])
        acc_plain[b] = cp / n
        acc_gated[b] = cg / n
        # retention among full-correct
        rc = sum(1 for i in full_correct_ids if b in by_id[i] and by_id[i][b]["correct_plain"])
        retain[b] = (rc, n_full)

    A_full = acc_plain[1.0]

    # rho (plain): full wrong and some b<1 correct
    rho_num = 0
    for i in ids:
        r1 = by_id[i].get(1.0)
        if r1 is None:
            continue
        if not r1["correct_plain"]:
            if any(by_id[i][b]["correct_plain"] for b in budgets if b < 1.0 and b in by_id[i]):
                rho_num += 1
    rho = rho_num / N

    # drop / gate (one drop per id)
    drops = [by_id[i][1.0]["drop"] for i in ids if 1.0 in by_id[i]]
    mean_D = st.mean(drops)
    std_D = st.pstdev(drops)
    below = sum(1 for d in drops if d < TAU)
    gate_open_frac = 1 - below / len(drops)

    Ts = [by_id[i][1.0]["T"] for i in ids if 1.0 in by_id[i]]

    # catastrophe: retention at smallest budget
    b_min = min(budgets)
    ret_lo = retain[b_min]
    ret_frac_lo = ret_lo[0] / ret_lo[1] if ret_lo[1] else float("nan")

    # empirical classification
    # capacity-bound: accuracy destroyed (retention collapses) + rho ~ 0
    # dilution-prone: rho >= 0.05
    # evict-robust: accuracy roughly flat + rho ~ 0
    acc_drop_ratio = (A_full - acc_plain[b_min]) / A_full if A_full > 0 else float("nan")
    collapse = (n_full >= 5) and (ret_frac_lo <= 0.25)  # >=75% of correct destroyed
    if rho >= 0.05:
        emp_class = "dilution-prone"
    elif collapse:
        emp_class = "capacity-bound"
    elif A_full >= 0.05 and not collapse:
        emp_class = "evict-robust"
    else:
        emp_class = "untestable(floor)" if A_full < 0.05 else "ambiguous"

    # gate prediction: gate mostly closed (do-not-evict) => predicts capacity-bound
    # gate mostly open (evict) => predicts dilution-prone/evict-robust
    if gate_open_frac >= 0.5:
        gate_pred = "evict (dilution/robust)"
    else:
        gate_pred = "do-not-evict (capacity-bound)"

    # was gate right?
    if emp_class == "capacity-bound":
        gate_right = gate_open_frac < 0.5
    elif emp_class in ("dilution-prone", "evict-robust"):
        gate_right = gate_open_frac >= 0.5
    else:
        gate_right = None

    return {
        "task": task, "path": path, "N": N, "n_full": n_full,
        "A_full": A_full, "acc_plain": acc_plain, "acc_gated": acc_gated,
        "retain": retain, "budgets": budgets, "rho": rho,
        "mean_D": mean_D, "std_D": std_D, "gate_open_frac": gate_open_frac,
        "T_med": int(st.median(Ts)), "T_min": min(Ts), "T_max": max(Ts),
        "ret_frac_lo": ret_frac_lo, "b_min": b_min, "acc_drop_ratio": acc_drop_ratio,
        "emp_class": emp_class, "gate_pred": gate_pred, "gate_right": gate_right,
    }


def main():
    paths = sys.argv[1:]
    stats = [analyze(p) for p in paths]

    md = []
    md.append("# Realistic-workload partition + gate test (Qwen2.5-14B-Instruct, LongBench)\n")
    md.append(f"- tau = {TAU}; budgets {stats[0]['budgets']}; two_pass; obs_window 32, n_sink 4, top_k 32.")
    md.append("- Plain = SnapKV-style eviction always on. Gated = evict only if gate open (D>=tau).")
    md.append("- Metric: any-in substring match (gold answer/line appears in prediction).\n")

    md.append("## Summary table\n")
    md.append("| Task | N | T_med | A_full | acc@0.5 | acc@0.125 | acc@0.0625 | full-correct retained @0.0625 | rho | mean D | gate-open frac | gate predicts | empirical class | gate correct? |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in stats:
        bmin = s["b_min"]
        rl = s["retain"][bmin]
        md.append(
            f"| {s['task']} | {s['N']} | {s['T_med']} | {s['A_full']:.3f} | "
            f"{s['acc_plain'].get(0.5, float('nan')):.3f} | {s['acc_plain'].get(0.125, float('nan')):.3f} | "
            f"{s['acc_plain'][bmin]:.3f} | {rl[0]}/{rl[1]} ({s['ret_frac_lo']:.2f}) | "
            f"{s['rho']:.3f} | {s['mean_D']:+.4f} | {s['gate_open_frac']:.3f} | "
            f"{s['gate_pred']} | {s['emp_class']} | "
            f"{'YES' if s['gate_right'] else ('NO' if s['gate_right'] is False else 'n/a')} |"
        )
    md.append("")

    for s in stats:
        md.append(f"## {s['task']}  (N={s['N']}, T {s['T_min']}-{s['T_max']}, median {s['T_med']})\n")
        md.append(f"- A_full = {s['A_full']:.3f}  ({s['n_full']} full-KV-correct inputs)")
        md.append("")
        md.append("| budget | plain acc | gated acc | full-correct retained (plain) |")
        md.append("|---:|---:|---:|---:|")
        for b in s["budgets"]:
            rc, nf = s["retain"][b]
            md.append(f"| {b} | {s['acc_plain'][b]:.3f} | {s['acc_gated'][b]:.3f} | {rc}/{nf} |")
        md.append("")
        md.append(f"- rho = {s['rho']:.3f}   (Pr[full-KV wrong AND some b<1.0 correct])")
        md.append(f"- mean head-agreement drop D = {s['mean_D']:+.4f} (std {s['std_D']:.4f}); gate-open fraction at tau={TAU}: {s['gate_open_frac']:.3f}")
        md.append(f"- accuracy loss full->b_min: {s['acc_drop_ratio']*100:.0f}%; retention of full-correct at b_min: {s['ret_frac_lo']:.2f}")
        md.append(f"- **Empirical class: {s['emp_class']}**; gate predicts: {s['gate_pred']}; "
                  f"gate correct: {'YES' if s['gate_right'] else ('NO' if s['gate_right'] is False else 'n/a')}")
        md.append("")

    with open(OUT_MD, "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {OUT_MD}")


if __name__ == "__main__":
    main()
