"""Post-hoc τ sensitivity for gated eviction.

Uses existing gated_eviction.py output (which always saves the plain
evicted pred AND drop per example, plus the plain pred at budget=1.0
which is the full-KV outcome). For any candidate τ' we can compute the
gated decision: if drop >= τ', use evicted pred; else, use full-KV pred.

Outputs gated mean accuracy as a function of τ at each budget.
"""
import argparse
import json
from collections import defaultdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--taus", default="0.00,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.10,0.12,0.15,0.20")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    rows = [json.loads(l) for l in open(args.inp)]
    taus = [float(t) for t in args.taus.split(",")]

    # Build full_KV outcome per example: plain pred at budget=1.0
    full_kv = {}
    for r in rows:
        if r["budget"] == 1.0:
            full_kv[r["id"]] = r["correct_plain"]

    by_b_tau = defaultdict(lambda: defaultdict(list))
    budgets = sorted({r["budget"] for r in rows}, reverse=True)

    for r in rows:
        if r["budget"] == 1.0:
            continue
        id_, b = r["id"], r["budget"]
        evicted_ok = r["correct_plain"]
        full_ok = full_kv.get(id_, False)
        drop = r["drop"]
        for tau in taus:
            gated_ok = evicted_ok if drop >= tau else full_ok
            by_b_tau[b][tau].append(gated_ok)

    plain_by_b = defaultdict(list)
    for r in rows:
        if r["budget"] < 1.0:
            plain_by_b[r["budget"]].append(r["correct_plain"])

    lines = []
    lines.append(f"# Tau sensitivity: {args.inp}\n")
    lines.append("Gated mean accuracy as a function of tau, at each budget. "
                 "Plain SnapKV at budget=b for reference.\n")
    header_taus = " | ".join(f"τ={t:g}" for t in taus)
    lines.append(f"| budget | plain | {header_taus} |")
    lines.append("|---:|---:|" + "---:|" * len(taus))
    for b in budgets:
        if b == 1.0:
            continue
        plain_acc = sum(plain_by_b[b]) / max(1, len(plain_by_b[b]))
        gated_accs = [
            sum(by_b_tau[b][t]) / max(1, len(by_b_tau[b][t]))
            for t in taus
        ]
        row_cells = " | ".join(f"{a:.3f}" for a in gated_accs)
        lines.append(f"| {b:.4g} | {plain_acc:.3f} | {row_cells} |")
    lines.append("")
    # Best τ per budget
    lines.append("\n## Best τ per budget (max gated mean accuracy)\n")
    lines.append(f"| budget | plain | best τ | best gated | Δ |")
    lines.append("|---:|---:|---:|---:|---:|")
    for b in budgets:
        if b == 1.0:
            continue
        plain_acc = sum(plain_by_b[b]) / max(1, len(plain_by_b[b]))
        gated_accs = [
            (t, sum(by_b_tau[b][t]) / max(1, len(by_b_tau[b][t])))
            for t in taus
        ]
        best = max(gated_accs, key=lambda x: x[1])
        lines.append(f"| {b:.4g} | {plain_acc:.3f} | {best[0]:g} | {best[1]:.3f} | {best[1]-plain_acc:+.3f} |")

    out = "\n".join(lines)
    print(out)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
