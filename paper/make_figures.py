"""Generate the 3 paper figures referenced in main.tex.

Figures:
  1. figs/method_agnostic_matrix.pdf — grouped bar chart of Δ per (model, base evictor).
  2. figs/scaling_formula.pdf        — scatter of ρ vs (1 - A_full) showing the linear-ish relationship.
  3. figs/mistral_niah_showcase.pdf  — plain vs gated accuracy curves on Mistral 4K NIAH-MK3.

Numbers traced from:
  - experiments/results/method_agnostic_matrix.md
  - experiments/results/gating_consolidated_report.md
  - experiments/results/gated_4k_mistral7b.jsonl (per-task)
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Match a clean ICLR-ish style.
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": ":",
    "lines.linewidth": 1.4,
    "patch.linewidth": 0.6,
    "legend.frameon": False,
    "savefig.bbox": "tight",
})

FIGS = Path(__file__).parent / "figs"
FIGS.mkdir(exist_ok=True)

RESULTS = Path(__file__).parent.parent / "experiments" / "results"


# ============================================================
# Figure 1: 4x4 method-agnostic matrix as grouped bars of Δ
# ============================================================
def figure_matrix():
    models = ["Qwen 1.5B", "Qwen 3B", "Qwen 14B", "Mistral 7B"]
    policies = ["SnapKV", "H2O", "StreamingLLM", "PyramidKV"]
    # rows = models, columns = policies; values are Δ (gated - plain)
    deltas = np.array([
        [0.121, 0.162, 0.163, 0.152],  # Qwen 1.5B
        # Qwen 3B SnapKV: post-hoc exact re-evaluation at tau=0.07
        # (run executed at tau=0.04); see paper App. repro notes.
        [0.259, 0.303, 0.430, 0.329],  # Qwen 3B
        [0.143, 0.200, 0.250, 0.228],  # Qwen 14B
        [0.187, 0.236, 0.263, 0.231],  # Mistral 7B
    ])
    n_models, n_policies = deltas.shape

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    x = np.arange(n_models)
    width = 0.18
    colors = ["#4477AA", "#EE6677", "#228833", "#CCBB44"]
    for i, p in enumerate(policies):
        offsets = (i - (n_policies - 1) / 2) * width
        bars = ax.bar(x + offsets, deltas[:, i] * 100, width,
                      label=p, color=colors[i], edgecolor="black", linewidth=0.4)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                    f"{b.get_height():.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel(r"Gated $-$ plain ($\Delta$, pp, mean over budgets $b<1$)")
    ax.set_title(r"Method-agnostic gating: every cell positive, grand mean $+22.9$pp",
                 fontsize=10)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=9)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylim(0, max(deltas.flatten()) * 100 * 1.15)
    fig.savefig(FIGS / "method_agnostic_matrix.pdf")
    fig.savefig(FIGS / "method_agnostic_matrix.png", dpi=150)
    print(f"wrote {FIGS / 'method_agnostic_matrix.pdf'}")


# ============================================================
# Figure 2: scatter of ρ vs (1 - A_full), color-coded by partition
# ============================================================
def figure_scaling():
    # (label, A_full, rho, regime, marker)
    # Numbers from h1_consolidated_report.md and gating_consolidated_report.md.
    points = [
        # Dilution-prone (D)
        ("Qwen1.5B 4K VT",        0.82, 0.06, "D"),
        ("Qwen1.5B 16K VT",       0.25, 0.19, "D"),
        ("Qwen3B 4K VT",          1.00, 0.00, "D"),
        ("Qwen3B 16K VT",         0.91, 0.05, "D"),
        ("Qwen1.5B 4K QA_1",      0.73, 0.08, "D"),
        ("Qwen1.5B 16K QA_1",     0.65, 0.14, "D"),
        ("Qwen3B 16K QA_1",       0.69, 0.08, "D"),
        ("Qwen1.5B 4K FWE",       0.22, 0.06, "D"),
        ("Qwen3B 16K FWE",        0.38, 0.32, "D"),
        ("Qwen1.5B 4K niah_mv",   0.30, 0.07, "D"),
        ("Qwen3B 16K niah_mv",    0.60, 0.13, "D"),
        ("Mistral 4K VT",         0.82, 0.10, "D"),
        ("Mistral 16K VT",        0.62, 0.12, "D"),
        # Capacity-bound (C)
        ("Qwen1.5B 4K NIAH-MK3",  0.65, 0.01, "C"),
        ("Qwen1.5B 16K NIAH-MK3", 0.32, 0.00, "C"),
        ("Qwen3B 4K NIAH-MK3",    0.93, 0.00, "C"),
        ("Qwen3B 16K NIAH-MK3",   0.26, 0.00, "C"),
        ("Mistral 4K NIAH-MK3",   0.99, 0.01, "C"),
        ("Mistral 16K NIAH-MK3",  0.68, 0.02, "C"),
    ]
    headroom = np.array([1 - p[1] for p in points])
    rhos = np.array([p[2] for p in points])
    regimes = np.array([p[3] for p in points])

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    d_mask = regimes == "D"
    c_mask = regimes == "C"

    ax.scatter(headroom[d_mask], rhos[d_mask], s=42, color="#4477AA", marker="o",
               edgecolor="black", linewidth=0.4, label=r"Dilution-prone $\mathcal{D}$", zorder=3)
    ax.scatter(headroom[c_mask], rhos[c_mask], s=42, color="#EE6677", marker="X",
               edgecolor="black", linewidth=0.4, label=r"Capacity-bound $\mathcal{C}$", zorder=3)

    # Linear fit to dilution-prone subset to illustrate the bound.
    if d_mask.sum() >= 3:
        x_d = headroom[d_mask]
        y_d = rhos[d_mask]
        # Force fit through origin: rho ≈ slope * headroom.
        slope = float(np.sum(x_d * y_d) / np.sum(x_d * x_d))
        xs = np.linspace(0, 1, 50)
        label = (r"OLS fit on $\mathcal{D}$: $\rho \approx "
                 + f"{slope:.2f}" + r"\cdot (1{-}A_{\rm full})$")
        ax.plot(xs, slope * xs, color="#4477AA", alpha=0.55, linewidth=1.0, linestyle="--",
                label=label)

    ax.set_xlabel(r"Headroom $\;1 - A_{\mathrm{full}}(M, T)$")
    ax.set_ylabel(r"Per-input recovery rate $\rho_{\mathrm{KV}}$")
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(-0.02, 0.4)
    ax.set_title(r"$\rho$ tracks the scaling formula on $\mathcal{D}$; $\rho\!\approx\!0$ on $\mathcal{C}$",
                 fontsize=10)
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(FIGS / "scaling_formula.pdf")
    fig.savefig(FIGS / "scaling_formula.png", dpi=150)
    print(f"wrote {FIGS / 'scaling_formula.pdf'}")


# ============================================================
# Figure 3: plain vs gated SnapKV on Mistral 4K NIAH-MK3
# ============================================================
def figure_showcase():
    path = RESULTS / "gated_4k_mistral7b.jsonl"
    if not path.exists():
        print(f"missing {path}; skipping figure 3")
        return
    from collections import defaultdict
    rows = [json.loads(l) for l in open(path) if json.loads(l)["task"] == "niah_multikey_3"]
    by_b = defaultdict(lambda: {"plain": [], "gated": []})
    for r in rows:
        by_b[r["budget"]]["plain"].append(r["correct_plain"])
        by_b[r["budget"]]["gated"].append(r["correct_gated"])

    budgets = sorted(by_b)
    plain_acc = [sum(by_b[b]["plain"]) / len(by_b[b]["plain"]) for b in budgets]
    gated_acc = [sum(by_b[b]["gated"]) / len(by_b[b]["gated"]) for b in budgets]

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(budgets, plain_acc, "o-", color="#EE6677", label="Plain SnapKV", markersize=6, linewidth=1.5)
    ax.plot(budgets, gated_acc, "s-", color="#4477AA", label="Gated SnapKV (ours)", markersize=6, linewidth=1.5)
    ax.fill_between(budgets, plain_acc, gated_acc, alpha=0.12, color="#4477AA",
                    label="Recovery $\\Delta$")
    ax.set_xlabel("Budget ratio $b$ (kept fraction of KV)")
    ax.set_ylabel("Accuracy on NIAH-MultiKey-3")
    ax.set_title(r"Mistral-7B-v0.3, RULER 4K NIAH-MK3 ($N{=}100$): gated holds $\sim 89\%$ at every budget",
                 fontsize=9.5)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlim(0, 1.05)
    ax.invert_xaxis()  # show aggressive eviction on the right side
    ax.legend(loc="lower left", fontsize=9)
    # Annotate the largest gap
    if plain_acc and gated_acc:
        gaps = [g - p for g, p in zip(gated_acc, plain_acc)]
        i = int(np.argmax(gaps))
        ax.annotate(f"$\\Delta = +{gaps[i]:.2f}$",
                    xy=(budgets[i], (plain_acc[i] + gated_acc[i]) / 2),
                    xytext=(0.3, 0.5),
                    fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.6))
    fig.savefig(FIGS / "mistral_niah_showcase.pdf")
    fig.savefig(FIGS / "mistral_niah_showcase.png", dpi=150)
    print(f"wrote {FIGS / 'mistral_niah_showcase.pdf'}")


if __name__ == "__main__":
    figure_matrix()
    figure_scaling()
    figure_showcase()
