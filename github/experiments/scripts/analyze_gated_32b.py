"""Analyze 32B gated-eviction run (gated_4k_qwen25_32b.jsonl).

Beyond analyze_gated.py, this adds:
  - per-task mean/median drops + headroom (1 - A_full) for the saturation check
  - post-hoc gated accuracy at arbitrary tau via the identity:
      gated(tau) = plain(b) if drop >= tau else full-KV outcome (plain @ b=1.0)
  - pooled z-score threshold tau_z = mu - 0.69*sigma (theta_z from
    normalized_predictor.md, fit on Qwen/Mistral calibration cells)
"""
import argparse
import json
import statistics
from collections import defaultdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--run_tau", type=float, default=0.07)
    p.add_argument("--theta_z", type=float, default=-0.69)
    args = p.parse_args()

    rows = [json.loads(l) for l in open(args.inp)]
    if not rows:
        print("no rows")
        return

    tasks = sorted({r["task"] for r in rows})
    budgets = sorted({r["budget"] for r in rows}, reverse=True)
    evict_budgets = [b for b in budgets if b < 1.0]

    # Index: (id, task) -> per-budget records; and full-KV outcome per example.
    by_ex = defaultdict(dict)   # (task, id) -> {budget: rec}
    for r in rows:
        by_ex[(r["task"], r["id"])][r["budget"]] = r

    # Drop is per-example (same across budgets).
    ex_drop = {}       # (task, id) -> drop
    ex_full_ok = {}    # (task, id) -> full-KV correctness (plain @ b=1.0)
    for key, recs in by_ex.items():
        anyrec = next(iter(recs.values()))
        ex_drop[key] = anyrec["drop"]
        if 1.0 in recs:
            ex_full_ok[key] = recs[1.0]["correct_plain"]
        else:
            ex_full_ok[key] = None

    def gated_acc_at_tau(tau, task_filter=None):
        """Post-hoc identity: per (example, eviction budget):
        gated = plain(b) if drop >= tau else full-KV outcome."""
        num, den = 0, 0
        for key, recs in by_ex.items():
            if task_filter and key[0] != task_filter:
                continue
            full_ok = ex_full_ok[key]
            for b in evict_budgets:
                if b not in recs:
                    continue
                if ex_drop[key] >= tau:
                    ok = recs[b]["correct_plain"]
                else:
                    ok = full_ok if full_ok is not None else recs[b]["correct_gated"]
                num += int(ok)
                den += 1
        return num / den if den else float("nan"), den

    def plain_mean(task_filter=None):
        num, den = 0, 0
        for key, recs in by_ex.items():
            if task_filter and key[0] != task_filter:
                continue
            for b in evict_budgets:
                if b in recs:
                    num += int(recs[b]["correct_plain"])
                    den += 1
        return num / den if den else float("nan")

    lines = []
    lines.append("# Gated eviction at 32B: Qwen2.5-32B-Instruct, RULER 4K")
    lines.append("")
    lines.append(f"- input: {args.inp}")
    lines.append(f"- tasks: {tasks}")
    lines.append(f"- budgets: {budgets}")
    lines.append(f"- run tau: {args.run_tau} (score_policy snapkv, two-pass sdpa)")
    lines.append("")

    # (a) per-task accuracy tables
    lines.append("## (a) Per-task accuracy: plain vs gated (tau=0.07, as-run)")
    lines.append("")
    task_full = {}
    for task in tasks:
        keys = [k for k in by_ex if k[0] == task]
        n = len(keys)
        opens = [1 for k in keys if ex_drop[k] >= args.run_tau]
        a_full_vals = [ex_full_ok[k] for k in keys if ex_full_ok[k] is not None]
        a_full = sum(a_full_vals) / len(a_full_vals) if a_full_vals else float("nan")
        task_full[task] = a_full
        lines.append(f"### task = {task}")
        lines.append(f"- N = {n}; A_full = {a_full:.3f}; "
                     f"gate-open fraction @ tau={args.run_tau}: {sum(opens)/n:.3f}")
        lines.append("| budget | plain acc | gated acc | delta |")
        lines.append("|---:|---:|---:|---:|")
        for b in budgets:
            pv = [by_ex[k][b]["correct_plain"] for k in keys if b in by_ex[k]]
            gv = [by_ex[k][b]["correct_gated"] for k in keys if b in by_ex[k]]
            pa = sum(pv) / len(pv)
            ga = sum(gv) / len(gv)
            lines.append(f"| {b:.4g} | {pa:.3f} | {ga:.3f} | {ga-pa:+.3f} |")
        lines.append("")

    # pooled mixed-suite table
    lines.append("### Mixed suite (pooled)")
    lines.append("| budget | plain mean | gated mean | delta | n_gate_open / N |")
    lines.append("|---:|---:|---:|---:|---:|")
    for b in budgets:
        pv, gv, ov = [], [], []
        for key, recs in by_ex.items():
            if b in recs:
                pv.append(recs[b]["correct_plain"])
                gv.append(recs[b]["correct_gated"])
                ov.append(recs[b]["gate_open"])
        pa, ga = sum(pv)/len(pv), sum(gv)/len(gv)
        lines.append(f"| {b:.4g} | {pa:.3f} | {ga:.3f} | {ga-pa:+.3f} | {sum(ov)}/{len(ov)} |")
    lines.append("")

    g07, _ = gated_acc_at_tau(args.run_tau)
    pm = plain_mean()
    lines.append(f"Headline (mean over eviction budgets, mixed suite): "
                 f"plain {pm:.3f}, gated@tau={args.run_tau} {g07:.3f}, "
                 f"**delta {g07-pm:+.3f}**")
    lines.append("")

    # (b) partition / saturation check
    lines.append("## (b) Partition check at 32B: per-task drops and headroom")
    lines.append("")
    lines.append("| task | N | mean drop | median drop | min | max | A_full | headroom (1-A_full) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    task_mean_drop = {}
    for task in tasks:
        d = [ex_drop[k] for k in by_ex if k[0] == task]
        task_mean_drop[task] = statistics.mean(d)
        lines.append(f"| {task} | {len(d)} | {statistics.mean(d):+.4f} | "
                     f"{statistics.median(d):+.4f} | {min(d):+.4f} | {max(d):+.4f} | "
                     f"{task_full[task]:.3f} | {1-task_full[task]:.3f} |")
    lines.append("")
    order = sorted(task_mean_drop, key=task_mean_drop.get)
    mk3_smallest = order[0] == "niah_multikey_3"
    lines.append(f"- Drop ordering (ascending): {' < '.join(order)}")
    lines.append(f"- NIAH-MK3 smallest mean drop: **{'YES' if mk3_smallest else 'NO'}**")
    lines.append("")

    # (c) z-score threshold
    all_drops = list(ex_drop.values())
    mu = statistics.mean(all_drops)
    sigma = statistics.pstdev(all_drops)
    tau_z = mu + args.theta_z * sigma
    lines.append("## (c) z-score threshold check")
    lines.append("")
    lines.append(f"- pooled drops: N = {len(all_drops)}, mu = {mu:+.4f}, sigma = {sigma:.4f}")
    lines.append(f"- tau_z = mu + ({args.theta_z})*sigma = **{tau_z:+.4f}**")
    lines.append("")
    gz, _ = gated_acc_at_tau(tau_z)
    lines.append("| policy | mixed-suite mean acc (eviction budgets) | delta vs plain |")
    lines.append("|---|---:|---:|")
    lines.append(f"| plain (always evict) | {pm:.3f} | — |")
    lines.append(f"| gated @ tau=0.07 (fixed) | {g07:.3f} | {g07-pm:+.3f} |")
    lines.append(f"| gated @ tau_z={tau_z:+.4f} | {gz:.3f} | {gz-pm:+.3f} |")
    lines.append("")
    lines.append("Per-task gate-open fractions at both thresholds:")
    lines.append("| task | open @ tau=0.07 | open @ tau_z | gated acc @0.07 | gated acc @tau_z | plain acc |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for task in tasks:
        keys = [k for k in by_ex if k[0] == task]
        o07 = sum(1 for k in keys if ex_drop[k] >= args.run_tau) / len(keys)
        oz = sum(1 for k in keys if ex_drop[k] >= tau_z) / len(keys)
        g07t, _ = gated_acc_at_tau(args.run_tau, task_filter=task)
        gzt, _ = gated_acc_at_tau(tau_z, task_filter=task)
        pt = plain_mean(task_filter=task)
        lines.append(f"| {task} | {o07:.2f} | {oz:.2f} | {g07t:.3f} | {gzt:.3f} | {pt:.3f} |")
    lines.append("")

    out = "\n".join(lines)
    print(out)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
