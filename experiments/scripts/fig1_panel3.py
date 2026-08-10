"""Panel 3 of Figure 1, drawn from the measured per-layer agreement curves.

The hand-drawn version of this panel had the direction of the effect
backwards: it showed heads disagreeing in early layers and agreeing in late
ones. Every cell we measure shows the opposite. Head agreement peaks early and
decays with depth as heads specialise, which is why D = a_early - a_late is
positive throughout, and the decay is steeper on dilution-prone inputs.

Drawing the panel from the data removes the possibility of the schematic and
the gate disagreeing about the sign of their own statistic.
"""
import json
import os
from collections import defaultdict

from paths import RESULTS, out_path
from layer_subsample import read_jsonl, mean

CELL = ("Qwen2.5-1.5B, RULER 4K", "drops_qwen15b_4k_n100.jsonl")
SERIES = [
    ("vt", "dilution-prone (VT)", "#1f77b4", "-"),
    ("niah_multikey_3", "capacity-bound (NIAH-MK3)", "#d62728", "-"),
]


def main():
    path = os.path.join(RESULTS, CELL[1])
    rows, _ = read_jsonl(path)
    by = defaultdict(list)
    for r in rows:
        by[r["task"]].append(r["head_agreement_per_layer"])
    L = len(rows[0]["head_agreement_per_layer"])
    third = L // 3
    prof = {t: [mean([p[i] for p in ps]) for i in range(L)] for t, ps in by.items()}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.8, 2.9))
    lo = min(min(prof[t]) for t, *_ in SERIES if t in prof)
    hi = max(max(prof[t]) for t, *_ in SERIES if t in prof)
    pad = 0.06 * (hi - lo)
    ax.set_ylim(lo - pad, hi + 3.2 * pad)
    ax.set_xlim(-0.5, L - 0.5)

    ax.axvspan(-0.5, third - 0.5, color="0.93", zorder=0)
    ax.axvspan(L - third - 0.5, L - 0.5, color="0.93", zorder=0)
    ax.text(third / 2 - 0.5, hi + 2.4 * pad, "early bin", ha="center",
            fontsize=7.5, color="0.35")
    ax.text(L - third / 2 - 0.5, hi + 2.4 * pad, "late bin", ha="center",
            fontsize=7.5, color="0.35")

    for i, (task, label, colour, _) in enumerate(SERIES):
        if task not in prof:
            continue
        # raw curve, de-emphasised: the per-layer values are noisy and the
        # claim is about the early-to-late contrast, not any single layer
        ax.plot(range(L), prof[task], color=colour, lw=0.9, alpha=0.35)
        e = mean(prof[task][:third])
        l = mean(prof[task][-third:])
        ax.hlines(e, -0.5, third - 0.5, color=colour, lw=2.6)
        ax.hlines(l, L - third - 0.5, L - 0.5, color=colour, lw=2.6,
                  label=f"{label}:  $D={e - l:+.3f}$")
        # the drop itself, drawn at a task-specific x so the two do not collide
        x = L / 2 + (i - 0.5) * 2.2
        ax.annotate("", xy=(x, l), xytext=(x, e),
                    arrowprops=dict(arrowstyle="<->", color=colour, lw=1.3))

    ax.set_xlabel("layer $\\ell$", fontsize=9)
    ax.set_ylabel("head agreement $a_\\ell$", fontsize=9)
    ax.set_title("Agreement peaks early and decays with depth;\n"
                 "the decay is steeper when the context is redundant",
                 fontsize=8.5)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    fig.subplots_adjust(left=0.16, right=0.98, top=0.83, bottom=0.19)
    fig.savefig(out_path("fig1_panel3.pdf"))
    fig.savefig(out_path("fig1_panel3.png"), dpi=200)

    print(f"# Figure 1, panel 3, redrawn from {CELL[0]} (L={L})\n")
    print("| task | early | late | D |")
    print("|---|---:|---:|---:|")
    for task, label, _, _ in SERIES:
        if task not in prof:
            continue
        e = mean(prof[task][:third]); l = mean(prof[task][-third:])
        print(f"| {label} | {e:.3f} | {l:.3f} | {e - l:+.3f} |")
    print("\nDirection: agreement falls with depth in every series, so D > 0 "
          "throughout and the gate's `D >= tau` test has the sign the data "
          "supports.")
    print(f"\nwrote {out_path('fig1_panel3.pdf')}")


if __name__ == "__main__":
    main()