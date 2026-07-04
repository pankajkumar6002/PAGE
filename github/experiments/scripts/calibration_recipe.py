"""One-shot τ calibration recipe.

For a new (model, context) cell, calibrate τ with a 50-input pilot:
  1. Measure the drop on 20 inputs from a known capacity-bound task
     (NIAH-MultiKey-3 from RULER).
  2. Measure the drop on 20 inputs from a known dilution-prone task
     (VT from RULER).
  3. Compute τ = (mean_drop(NIAH-MK3) + mean_drop(VT)) / 2.

This replaces the hand-set τ=0.07 with a per-cell midpoint, which should fix
the Qwen3B-16K calibration shift and be neutral on all other cells.

Verification: compare gated-SnapKV with τ_calibrated vs τ=0.07 on the mixed
suite at each (model, context). Recipe wins on Qwen3B-16K; matches elsewhere.

This script runs the verification using the EXISTING gated_eviction.py outputs
plus per_layer_agreement_*.jsonl files. No new sweeps required.
"""
import argparse
import json
import os
import sys
from collections import defaultdict


def load_drops_per_task(probe_path):
    """Return {task: list_of_drops} from a per_layer_agreement_*.jsonl probe output."""
    drops = defaultdict(list)
    with open(probe_path) as f:
        for line in f:
            r = json.loads(line)
            pl = r["head_agreement_per_layer"]
            L = len(pl)
            third = max(1, L // 3)
            early = sum(pl[:third]) / third
            late = sum(pl[L - third:]) / third
            drops[r["task"]].append(early - late)
    return drops


def calibrate_tau(probe_path, capacity_task="niah_multikey_3", dilution_task="vt"):
    """Apply the 20+20 midpoint calibration recipe."""
    drops = load_drops_per_task(probe_path)
    if capacity_task not in drops or dilution_task not in drops:
        return None
    c = sum(drops[capacity_task]) / len(drops[capacity_task])
    d = sum(drops[dilution_task]) / len(drops[dilution_task])
    return (c + d) / 2, c, d


def gated_acc_at_tau(gated_jsonl, tau):
    """Post-hoc evaluate gated mean accuracy at a given τ over the gated_eviction
    output (uses saved drop and saved full-KV outcome from budget=1.0)."""
    rows = [json.loads(l) for l in open(gated_jsonl)]
    full_kv = {}
    for r in rows:
        if r["budget"] == 1.0:
            full_kv[r["id"]] = r["correct_plain"]
    budgets = sorted({r["budget"] for r in rows}, reverse=True)
    by_b = defaultdict(list)
    for r in rows:
        if r["budget"] == 1.0:
            continue
        gated_ok = r["correct_plain"] if r["drop"] >= tau else full_kv.get(r["id"], False)
        by_b[r["budget"]].append(gated_ok)
    plain_by_b = defaultdict(list)
    for r in rows:
        if r["budget"] < 1.0:
            plain_by_b[r["budget"]].append(r["correct_plain"])
    eviction_budgets = [b for b in budgets if b < 1.0]
    if not eviction_budgets:
        return None, None
    plain_mean = sum(sum(plain_by_b[b]) for b in eviction_budgets) / sum(len(plain_by_b[b]) for b in eviction_budgets)
    gated_mean = sum(sum(by_b[b]) for b in eviction_budgets) / sum(len(by_b[b]) for b in eviction_budgets)
    return plain_mean, gated_mean


def main():
    base = "experiments/results"
    cells = [
        ("Qwen 1.5B 4K", f"{base}/per_layer_agreement.jsonl",
                          f"{base}/gated_4k_qwen15b.jsonl"),
        ("Qwen 3B 4K",   f"{base}/per_layer_agreement_qwen3b.jsonl",
                          f"{base}/gated_4k_qwen3b.jsonl"),
        ("Qwen 3B 16K",  f"{base}/per_layer_agreement_qwen3b_16k.jsonl",
                          f"{base}/gated_16k_qwen3b.jsonl"),
        ("Mistral 4K",   f"{base}/per_layer_agreement_mistral.jsonl",
                          f"{base}/gated_4k_mistral7b.jsonl"),
    ]

    print("# One-shot τ calibration recipe results\n")
    print(f"## Recipe: τ = (mean_drop(NIAH-MK3) + mean_drop(VT)) / 2 over 20-input pilots each\n")
    print(f"| Cell | drop(NIAH-MK3) | drop(VT) | τ_calibrated | τ_fixed=0.07 | gated@τ_cal | gated@τ=0.07 | plain | best lift |")
    print(f"|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cell, probe, gated in cells:
        if not (os.path.exists(probe) and os.path.exists(gated)):
            print(f"| {cell} | (missing data) |  |  |  |  |  |  |  |")
            continue
        cal = calibrate_tau(probe)
        if cal is None:
            print(f"| {cell} | (no NIAH-MK3 or VT in probe) |  |  |  |  |  |  |  |")
            continue
        tau_cal, c_drop, d_drop = cal
        plain_mean, gated_cal_mean = gated_acc_at_tau(gated, tau_cal)
        _, gated_07_mean = gated_acc_at_tau(gated, 0.07)
        lift_cal = gated_cal_mean - plain_mean
        lift_07 = gated_07_mean - plain_mean
        better = "calib" if lift_cal > lift_07 else "0.07"
        print(f"| {cell} | {c_drop:+.4f} | {d_drop:+.4f} | {tau_cal:+.4f} | 0.0700 "
              f"| {gated_cal_mean:.3f} ({lift_cal:+.3f}) | {gated_07_mean:.3f} ({lift_07:+.3f}) "
              f"| {plain_mean:.3f} | **{better}** |")


if __name__ == "__main__":
    main()
