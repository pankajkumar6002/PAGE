"""Tau sensitivity sweep for the gated-eviction threshold.

For each tau in a configurable list, derive a per-tau jsonl with the same
schema as gated_eviction.py output (so downstream tooling is identical) by
re-applying the gate decision (drop >= tau) to a base jsonl that already
recorded `drop`, the always-evict pred ("plain"), and the full-KV pred
(budget=1.0 record per id).

Why post-hoc: `drop` is a function of the prompt and the model's attentions
on prefill -- it does NOT depend on tau. The pred_plain at b<1 is the
always-evict outcome, also tau-independent. The full-KV pred is the b=1.0
plain record. So pred_gated for any tau' is:
    pred_gated = pred_plain          if drop >= tau'   (gate open => evict)
                 pred_at_b=1.0       otherwise         (gate closed => full KV)

This produces records bit-identical to running gated_eviction.py with --tau
tau' on the same model+data, modulo trivial floating-point drift in pred_*
strings (which are deterministic argmax decodes).

Outputs:
  - experiments/results/tau_sweep_tau<value>.jsonl  (per-tau jsonl)
  - experiments/results/tau_sensitivity.md          (summary table)
  - paper/figs/tau_sensitivity.pdf, .png            (Delta-vs-tau curve)
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Tasks the paper labels "dilution-prone" (gating should KEEP gate open / evict);
# vs "capacity-bound" (gating should CLOSE / keep full KV).
# NIAH-MK3 is the canonical capacity-bound retrieval-precision task.
CAPACITY_BOUND_TASKS = {"niah_multikey_3"}
DILUTION_PRONE_TASKS = {"vt", "fwe", "qa_1"}

TARGET_BUDGETS = (0.5, 0.25, 0.125, 0.0625)
TARGET_TASKS = ("niah_multikey_3", "vt", "fwe", "qa_1")


def load_base(base_path):
    rows = [json.loads(l) for l in open(base_path)]
    rows = [r for r in rows if r["task"] in TARGET_TASKS]
    # Full-KV pred per id is the budget=1.0 plain pred
    full_kv = {}
    for r in rows:
        if r["budget"] == 1.0:
            full_kv[r["id"]] = {
                "pred": r["pred_plain"],
                "correct": r["correct_plain"],
                "n_kept": r["n_kept_plain"],
            }
    return rows, full_kv


def rec_for_tau(r, tau, full_kv):
    """Re-derive the gated outcome for this row at the given tau."""
    drop = r["drop"]
    gate_open = drop >= tau
    if gate_open:
        pred_gated = r["pred_plain"]
        correct_gated = r["correct_plain"]
        n_kept_gated = r["n_kept_plain"]
    else:
        fk = full_kv[r["id"]]
        pred_gated = fk["pred"]
        correct_gated = fk["correct"]
        n_kept_gated = fk["n_kept"]
    out = dict(r)
    out["tau"] = tau
    out["gate_open"] = bool(gate_open)
    out["pred_gated"] = pred_gated
    out["correct_gated"] = bool(correct_gated)
    out["n_kept_gated"] = n_kept_gated
    out["score_policy"] = r.get("score_policy", "snapkv")
    return out


def write_per_tau_jsonls(base_rows, full_kv, taus, out_dir):
    paths = {}
    for tau in taus:
        path = Path(out_dir) / f"tau_sweep_tau{tau:g}.jsonl"
        with open(path, "w") as f:
            for r in base_rows:
                if r["budget"] not in TARGET_BUDGETS:
                    continue
                f.write(json.dumps(rec_for_tau(r, tau, full_kv)) + "\n")
        paths[tau] = path
    return paths


def metrics_for_tau(path):
    """Compute headline metrics from a per-tau jsonl.

    Returns dict with mean_delta, mean_gated_acc, mean_plain_acc, fp_rate, fn_rate.
    Aggregation:
      - mean Delta over budgets x tasks, equal weight per (task, budget) cell mean.
      - FP rate (capacity-bound MK3): gate-open fraction on MK3 examples.
      - FN rate (dilution-prone {vt, fwe, qa_1}): gate-closed fraction.
    """
    rows = [json.loads(l) for l in open(path)]
    # per (task, budget): mean plain, mean gated
    cell_plain = defaultdict(list)
    cell_gated = defaultdict(list)
    for r in rows:
        key = (r["task"], r["budget"])
        cell_plain[key].append(r["correct_plain"])
        cell_gated[key].append(r["correct_gated"])

    deltas = []
    plain_accs = []
    gated_accs = []
    for key in cell_plain:
        p = sum(cell_plain[key]) / len(cell_plain[key])
        g = sum(cell_gated[key]) / len(cell_gated[key])
        deltas.append(g - p)
        plain_accs.append(p)
        gated_accs.append(g)
    mean_delta = float(np.mean(deltas))
    mean_plain = float(np.mean(plain_accs))
    mean_gated = float(np.mean(gated_accs))

    # FP/FN: derived from gate_open, deduplicated per id (gate decision is
    # budget-independent for a given example).
    seen = {}
    for r in rows:
        seen[r["id"]] = (r["task"], r["gate_open"])
    fp_num = fp_den = 0
    fn_num = fn_den = 0
    for _id, (task, opened) in seen.items():
        if task in CAPACITY_BOUND_TASKS:
            fp_den += 1
            if opened:
                fp_num += 1
        elif task in DILUTION_PRONE_TASKS:
            fn_den += 1
            if not opened:
                fn_num += 1
    fp_rate = fp_num / fp_den if fp_den else 0.0
    fn_rate = fn_num / fn_den if fn_den else 0.0

    return {
        "mean_delta": mean_delta,
        "mean_plain_acc": mean_plain,
        "mean_gated_acc": mean_gated,
        "fp_rate": fp_rate,
        "fn_rate": fn_rate,
    }


def make_figure(taus, metrics, fig_dir):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
        "lines.linewidth": 1.4,
        "legend.frameon": False,
        "savefig.bbox": "tight",
    })
    deltas = np.array([metrics[t]["mean_delta"] for t in taus])
    fp = np.array([metrics[t]["fp_rate"] for t in taus])
    fn = np.array([metrics[t]["fn_rate"] for t in taus])
    gated = np.array([metrics[t]["mean_gated_acc"] for t in taus])

    delta_max = deltas.max()
    # Naive plateau: Delta within 1pp of the unconstrained max (can include
    # degenerate "high tau => gate-closed => use full KV always" regime).
    naive_mask = deltas >= (delta_max - 0.01)
    naive_taus = [t for t, m in zip(taus, naive_mask) if m]

    # Operational plateau: Delta within 1pp of Delta at the deployed tau=0.07,
    # AND the gate is doing meaningful work (FP + FN <= 0.10). This excludes
    # the high-tau regime where the gate closes on almost everything and the
    # method reduces to "always full KV", which is robust by construction but
    # not what the gating mechanism is for.
    try:
        idx_deploy = taus.index(0.07)
    except ValueError:
        idx_deploy = int(np.argmin(np.abs(np.array(taus) - 0.07)))
    delta_deploy = deltas[idx_deploy]
    work_mask = (fp + fn) <= 0.10
    plateau_mask = (np.abs(deltas - delta_deploy) <= 0.01) & work_mask
    plateau_taus = [t for t, m in zip(taus, plateau_mask) if m]
    if plateau_taus:
        plateau_lo, plateau_hi = min(plateau_taus), max(plateau_taus)
    else:
        plateau_lo = plateau_hi = 0.07

    fig, ax = plt.subplots(figsize=(4.4, 2.7))
    ax.plot(taus, deltas, marker="o", color="#1f4e79", label=r"$\Delta$ (gated $-$ plain)")
    if plateau_taus:
        ax.axvspan(plateau_lo, plateau_hi, alpha=0.13, color="#1f4e79",
                   label=(r"plateau ($|\Delta - \Delta_{\tau=0.07}|\leq 1$pp, "
                          r"FP$+$FN$\leq 0.1$)"))
    ax.axvline(0.07, ls="--", lw=0.9, color="0.3", label=r"$\tau = 0.07$ (deployed)")
    ax.set_xscale("log")
    ax.set_xlabel(r"gating threshold $\tau$")
    ax.set_ylabel(r"mean $\Delta$ across (task, budget)")
    ax.set_title(r"$\tau$ sensitivity (Qwen2.5-1.5B, RULER 4K)", fontsize=9)
    ax.set_ylim(-0.02, 0.30)
    ax.legend(loc="upper left", fontsize=7, handlelength=1.6)
    fig.tight_layout()
    pdf_path = Path(fig_dir) / "tau_sensitivity.pdf"
    png_path = Path(fig_dir) / "tau_sensitivity.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    return {
        "delta_max": float(delta_max),
        "delta_deploy": float(delta_deploy),
        "plateau_lo": plateau_lo,
        "plateau_hi": plateau_hi,
        "naive_plateau_lo": min(naive_taus),
        "naive_plateau_hi": max(naive_taus),
        "pdf": str(pdf_path),
        "png": str(png_path),
    }


def write_summary(taus, metrics, plateau_info, md_path):
    lines = []
    lines.append("# Tau sensitivity sweep")
    lines.append("")
    lines.append("Sweep of the gating threshold tau on Qwen2.5-1.5B-Instruct, "
                 "RULER 4K, mixed suite (NIAH-MK3 + VT + FWE + QA_1), "
                 "N=100 per task, budgets {0.5, 0.25, 0.125, 0.0625}, SnapKV.")
    lines.append("")
    lines.append("Delta = mean over (task, budget) cells of (gated_acc - plain_acc). "
                 "FP rate = fraction of capacity-bound MK3 examples on which the gate opens "
                 "(should be small). FN rate = fraction of dilution-prone (VT, FWE, QA_1) "
                 "examples on which the gate stays closed (should be small).")
    lines.append("")
    lines.append("| tau | Delta | FP (MK3) | FN (VT+FWE+QA_1) | gated mean acc | plain mean acc |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for t in taus:
        m = metrics[t]
        lines.append(
            f"| {t:g} | {m['mean_delta']:+.4f} | {m['fp_rate']:.3f} | "
            f"{m['fn_rate']:.3f} | {m['mean_gated_acc']:.3f} | {m['mean_plain_acc']:.3f} |"
        )
    lines.append("")
    lines.append("## Plateau analysis")
    lines.append("")
    lines.append(f"- Delta at deployed tau=0.07: {plateau_info['delta_deploy']:+.4f}")
    lines.append(f"- Unconstrained maximum Delta across the sweep: {plateau_info['delta_max']:+.4f}")
    lines.append(f"  (achieved at tau in [{plateau_info['naive_plateau_lo']:g}, "
                 f"{plateau_info['naive_plateau_hi']:g}], where the gate closes on "
                 "most dilution-prone examples too -- the system effectively reverts to "
                 "full KV, which is robust by construction but defeats the gating purpose.)")
    lines.append(f"- Operational plateau (|Delta - Delta_0.07| <= 1pp AND FP+FN <= 0.10): "
                 f"tau in [{plateau_info['plateau_lo']:g}, {plateau_info['plateau_hi']:g}]")
    in_plateau = plateau_info['plateau_lo'] <= 0.07 <= plateau_info['plateau_hi']
    plateau_str = "inside the plateau" if in_plateau else "outside the plateau"
    # Span of Delta inside the operational plateau
    plateau_deltas = [metrics[t]["mean_delta"] for t in taus
                      if plateau_info['plateau_lo'] <= t <= plateau_info['plateau_hi']]
    if len(plateau_deltas) >= 2:
        plateau_span_pp = (max(plateau_deltas) - min(plateau_deltas)) * 100
        lines.append(f"- Headline: Delta varies by {plateau_span_pp:.2f} pp across "
                     f"tau in [{plateau_info['plateau_lo']:g}, {plateau_info['plateau_hi']:g}]; "
                     f"tau = 0.07 sits {plateau_str}.")
    else:
        lines.append(f"- Headline: tau = 0.07 sits {plateau_str} "
                     f"[{plateau_info['plateau_lo']:g}, {plateau_info['plateau_hi']:g}].")
    lines.append("")
    lines.append(f"Figure: paper/figs/tau_sensitivity.pdf (+ .png)")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        default="experiments/results/gated_4k_qwen15b.jsonl",
        help="Base jsonl from gated_eviction.py with drop, pred_plain, and budget=1.0 records.",
    )
    ap.add_argument(
        "--taus",
        default="0.01,0.025,0.04,0.055,0.07,0.085,0.10,0.13,0.16,0.20",
    )
    ap.add_argument(
        "--out_dir",
        default="experiments/results",
    )
    ap.add_argument(
        "--fig_dir",
        default="paper/figs",
    )
    args = ap.parse_args()

    taus = [float(t) for t in args.taus.split(",")]
    base_rows, full_kv = load_base(args.base)

    # Verify coverage
    missing = [b for b in TARGET_BUDGETS if not any(r["budget"] == b for r in base_rows)]
    if missing:
        raise SystemExit(f"base jsonl missing budgets {missing}")
    if not full_kv:
        raise SystemExit("base jsonl missing budget=1.0 records (full-KV reference)")

    print(f"base: {args.base}  rows: {len(base_rows)}")
    print(f"taus: {taus}")

    paths = write_per_tau_jsonls(base_rows, full_kv, taus, args.out_dir)
    metrics = {t: metrics_for_tau(paths[t]) for t in taus}

    for t in taus:
        m = metrics[t]
        print(f"  tau={t:g}  Delta={m['mean_delta']:+.4f}  "
              f"FP_MK3={m['fp_rate']:.3f}  FN_dil={m['fn_rate']:.3f}  "
              f"gated_acc={m['mean_gated_acc']:.3f}")

    plateau_info = make_figure(taus, metrics, args.fig_dir)
    md_path = Path(args.out_dir) / "tau_sensitivity.md"
    write_summary(taus, metrics, plateau_info, md_path)
    print(f"figure: {plateau_info['pdf']}")
    print(f"summary: {md_path}")
    print(f"plateau: [{plateau_info['plateau_lo']:g}, {plateau_info['plateau_hi']:g}]; "
          f"Delta_max = {plateau_info['delta_max']:+.4f}")


if __name__ == "__main__":
    main()
