"""Generate per-input recovery distribution figures for Appendix B.

Outputs:
  - figs/recovery_histogram_by_task.pdf : faceted histogram of the per-input
    recovery indicator r(x) in {-1, 0, +1}, one panel per RULER task, pooled
    across (model, context) cells.
  - figs/recovery_histogram_by_cell.pdf : faceted bar chart of the fraction of
    inputs with r(x) > 0 per task, one panel per (model, context) cell.

Per-input recovery indicator is computed as
    r(x) = sign(max_b A_gated(b, x) - A_plain(b_max, x))
with b_max = 1.0 (full cache) and A in {0, 1} (binary correctness).
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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

PAPER = Path(__file__).parent
FIGS = PAPER / "figs"
FIGS.mkdir(exist_ok=True)
RESULTS = PAPER.parent / "experiments" / "results"

CELLS = [
    ("Qwen1.5B 4K",   "gated_4k_qwen15b.jsonl"),
    ("Qwen3B 4K",     "gated_4k_qwen3b.jsonl"),
    ("Mistral7B 4K",  "gated_4k_mistral7b.jsonl"),
    ("Qwen1.5B 16K",  "gated_16k_qwen15b.jsonl"),
    ("Qwen3B 16K",    "gated_16k_qwen3b.jsonl"),
    ("Mistral7B 16K", "gated_16k_mistral7b.jsonl"),
]

# Display labels for tasks present in the data.
TASK_LABEL = {
    "vt":              "VT",
    "fwe":             "FWE",
    "qa_1":            "QA_1",
    "niah_multikey_3": "NIAH-MK3",
}
TASK_ORDER = ["vt", "fwe", "qa_1", "niah_multikey_3"]

# Colour by partition: dilution-prone vs capacity-bound.
DILUTION = {"vt", "fwe", "qa_1"}
CAPACITY = {"niah_multikey_3"}
PARTITION_COLOR = {"D": "#4477AA", "C": "#EE6677"}


def load_cell(path):
    """Return list of rows from a jsonl file (empty if missing)."""
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def per_input_recovery(rows):
    """Compute r(x) in {-1, 0, +1} per (task, input_id) within one cell.

    r(x) = sign(max_b A_gated(b, x) - A_plain(1.0, x)).
    Inputs without a budget=1.0 plain row are skipped.
    """
    # Group by (task, input_id).
    plain_full = {}      # (task, id) -> A_plain at b=1.0
    gated_best = {}      # (task, id) -> max_b A_gated
    for r in rows:
        key = (r["task"], r["id"])
        if r["budget"] == 1.0:
            plain_full[key] = int(bool(r["correct_plain"]))
        gated_best[key] = max(gated_best.get(key, 0), int(bool(r["correct_gated"])))

    out = []  # (task, input_id, r)
    for key, p in plain_full.items():
        g = gated_best.get(key, 0)
        r_val = (1 if g > p else (-1 if g < p else 0))
        out.append((key[0], key[1], r_val))
    return out


def collect_all():
    """Return dict task -> list of (cell_label, r) tuples, pooled across cells."""
    by_task = defaultdict(list)
    by_cell = {}  # cell_label -> { task -> [r values] }
    for cell_label, fname in CELLS:
        rows = load_cell(RESULTS / fname)
        rec = per_input_recovery(rows)
        by_cell[cell_label] = defaultdict(list)
        for task, _id, r in rec:
            by_task[task].append((cell_label, r))
            by_cell[cell_label][task].append(r)
    return by_task, by_cell


# ============================================================
# Figure 1: histogram of recovery indicator, faceted by task
# ============================================================
def figure_by_task(by_task):
    tasks = [t for t in TASK_ORDER if t in by_task]
    n = len(tasks)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4, 2.4 * nrows),
                             sharey=False)
    axes = np.atleast_1d(axes).ravel()

    for ax, task in zip(axes, tasks):
        vals = np.array([r for _, r in by_task[task]], dtype=int)
        counts = [int((vals == k).sum()) for k in (-1, 0, 1)]
        partition = "D" if task in DILUTION else "C"
        color = PARTITION_COLOR[partition]
        bars = ax.bar([-1, 0, 1], counts, width=0.6, color=color,
                      edgecolor="black", linewidth=0.4)
        for b, c in zip(bars, counts):
            ax.text(b.get_x() + b.get_width() / 2,
                    b.get_height() + max(counts) * 0.015,
                    f"{c}", ha="center", va="bottom", fontsize=7)
        n_inputs = int(vals.size)
        frac_pos = counts[2] / n_inputs if n_inputs else 0.0
        ax.set_title(f"{TASK_LABEL[task]} ($n={n_inputs}$, $r{{>}}0$: {frac_pos:.0%})",
                     fontsize=9.5)
        ax.set_xticks([-1, 0, 1])
        ax.set_xticklabels([r"$-1$", r"$0$", r"$+1$"])
        ax.set_xlabel("recovery indicator $r(x)$")
        ax.set_ylabel("count")
        ax.set_ylim(0, max(counts) * 1.18 if max(counts) > 0 else 1)

    for ax in axes[len(tasks):]:
        ax.axis("off")

    fig.suptitle(r"Per-input recovery distribution by task "
                 r"(pooled across model$\times$context cells)",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGS / "recovery_histogram_by_task.pdf")
    fig.savefig(FIGS / "recovery_histogram_by_task.png", dpi=150)
    print(f"wrote {FIGS / 'recovery_histogram_by_task.pdf'}")


# ============================================================
# Figure 2: fraction recovered per task, faceted by (model, context)
# ============================================================
def figure_by_cell(by_cell):
    cells = [c for c, _ in CELLS]
    n = len(cells)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.4, 2.5 * nrows),
                             sharey=True)
    axes = np.atleast_1d(axes).ravel()

    tasks = TASK_ORDER

    for ax, cell in zip(axes, cells):
        fracs = []
        ns = []
        colors = []
        labels = []
        for task in tasks:
            rs = by_cell[cell].get(task, [])
            if not rs:
                fracs.append(0.0)
                ns.append(0)
            else:
                arr = np.array(rs, dtype=int)
                fracs.append(float((arr > 0).mean()))
                ns.append(int(arr.size))
            colors.append(PARTITION_COLOR["D" if task in DILUTION else "C"])
            labels.append(TASK_LABEL[task])

        x = np.arange(len(tasks))
        bars = ax.bar(x, fracs, color=colors, edgecolor="black", linewidth=0.4)
        for b, f, n_ in zip(bars, fracs, ns):
            if n_ == 0:
                txt = "n/a"
            else:
                txt = f"{f:.2f}"
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                    txt, ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        total_n = sum(ns)
        ax.set_title(f"{cell} ($n={total_n}$)", fontsize=9.5)
        ax.set_ylim(0, 0.55)
        ax.set_ylabel("fraction with $r(x)>0$")

    for ax in axes[len(cells):]:
        ax.axis("off")

    # One legend at the figure level.
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=PARTITION_COLOR["D"], edgecolor="black",
              label=r"Dilution-prone $\mathcal{D}$"),
        Patch(facecolor=PARTITION_COLOR["C"], edgecolor="black",
              label=r"Capacity-bound $\mathcal{C}$"),
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.04), ncol=2, frameon=False, fontsize=9)
    fig.suptitle(r"Fraction of inputs with strict gated$>$plain recovery, per cell",
                 fontsize=10, y=1.10)
    fig.tight_layout()
    fig.savefig(FIGS / "recovery_histogram_by_cell.pdf")
    fig.savefig(FIGS / "recovery_histogram_by_cell.png", dpi=150)
    print(f"wrote {FIGS / 'recovery_histogram_by_cell.pdf'}")


if __name__ == "__main__":
    by_task, by_cell = collect_all()
    # Diagnostics: print counts.
    print("Per-task pooled n and r distribution:")
    for task in TASK_ORDER:
        if task not in by_task:
            continue
        vals = np.array([r for _, r in by_task[task]], dtype=int)
        c = {k: int((vals == k).sum()) for k in (-1, 0, 1)}
        print(f"  {TASK_LABEL[task]:>8s}: n={vals.size:4d}  "
              f"r=-1:{c[-1]:3d}  r=0:{c[0]:3d}  r=+1:{c[1]:3d}")
    print("Per-cell n by task:")
    for cell, _ in CELLS:
        sizes = {TASK_LABEL[t]: len(by_cell[cell].get(t, [])) for t in TASK_ORDER}
        print(f"  {cell:>14s}: {sizes}")
    figure_by_task(by_task)
    figure_by_cell(by_cell)
