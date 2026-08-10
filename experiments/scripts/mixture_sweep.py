"""P1-6: how the headline Delta moves with the capacity-bound share of the workload.

The 4x4 matrix runs a mixed suite that is 1/4 NIAH-MK3. Since per-task Delta is
additive under reweighting,

    Delta(f) = f * Delta_MK3 + (1 - f) * Delta_rest,

and Delta_rest is exactly 0.000 in 8 of 16 cells, Delta(f) is close to linear
through the origin with slope Delta_MK3. This documents the concentration; it
does not remove it. Self-reporting is the point.

Writes a figure and a table. Uses matplotlib only if available.
"""
import os

from paths import MODELS, POLICIES, MK3, cell_path, out_path
from gatelib import eviction_rows, delta

FRACTIONS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.50, 0.75, 1.0]


def main():
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    cells = []
    for policy in POLICIES:
        for model_name, slug in MODELS:
            path = cell_path(slug, policy)
            if not os.path.exists(path):
                continue
            rows = eviction_rows(path, policy, slug)
            cells.append({
                "name": f"{policy} {model_name}",
                "mk3": delta(rows, lambda t: t == MK3),
                "rest": delta(rows, lambda t: t != MK3),
            })

    emit("# Mixture-fraction sweep: Delta as a function of the NIAH-MK3 share f")
    emit()
    emit("Delta(f) = f * Delta_MK3 + (1 - f) * Delta_rest, per cell, then averaged.")
    emit()
    emit("| f (MK3 share) | " + " | ".join(f"{f:.2f}" for f in FRACTIONS) + " |")
    emit("|---" * (len(FRACTIONS) + 1) + "|")

    means = []
    for f in FRACTIONS:
        vals = [f * c["mk3"] + (1 - f) * c["rest"] for c in cells]
        means.append(sum(vals) / len(vals))
    emit("| grand mean Delta | " + " | ".join(f"{m:+.3f}" for m in means) + " |")

    d0 = means[FRACTIONS.index(0.0)]
    d25 = means[FRACTIONS.index(0.25)]
    emit()
    emit(f"- Delta(f=0) = {d0:+.4f}, the no-MK3 grand mean")
    emit(f"- Delta(f=0.25) = {d25:+.4f}, the paper's mixed suite and its +22.9pp headline")
    emit(f"- slope between them: {(d25 - d0)/0.25:+.4f} per unit MK3 share")
    emit(f"- mean Delta_MK3 = {sum(c['mk3'] for c in cells)/len(cells):+.4f}, "
         f"mean Delta_rest = {sum(c['rest'] for c in cells)/len(cells):+.4f}")
    emit()
    emit("Reading: the headline number is a statement about a workload that is "
         "one quarter capacity-bound. On a workload with no capacity-bound "
         "inputs the gate is inert and Delta is near zero, which is the intended "
         "behaviour of a safeguard, not a failure of it.")

    emit()
    emit("## Per-cell endpoints")
    emit()
    emit("| cell | Delta_MK3 | Delta_rest |")
    emit("|---|---:|---:|")
    for c in cells:
        emit(f"| {c['name']} | {c['mk3']:+.3f} | {c['rest']:+.3f} |")

    with open(out_path("mixture_sweep.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fine = [i / 100 for i in range(0, 101)]
        fig, ax = plt.subplots(figsize=(4.2, 3.0))
        for c in cells:
            ax.plot(fine, [x * c["mk3"] + (1 - x) * c["rest"] for x in fine],
                    color="0.75", lw=0.8)
        ax.plot(fine, [sum(x * c["mk3"] + (1 - x) * c["rest"] for c in cells) / len(cells)
                       for x in fine], color="C0", lw=2.0, label="grand mean")
        ax.axvline(0.25, color="C3", ls="--", lw=1.0)
        ax.annotate("paper's mixed suite", xy=(0.25, 0.02), xytext=(0.30, 0.02),
                    fontsize=7, color="C3")
        ax.set_xlabel("NIAH-MK3 share of the workload, $f$")
        ax.set_ylabel(r"gating $\Delta$ (pp)")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        png = out_path("mixture_sweep.png")
        fig.savefig(png, dpi=200)
        fig.savefig(out_path("mixture_sweep.pdf"))
        print(f"\nwrote {png}")
    except ImportError:
        print("\nmatplotlib not available, skipped the figure")

    ok = abs(d0 - 0.045) < 1.5e-3 and abs(d25 - 0.229) < 1e-3
    if not ok:
        print(f"\nFAIL: endpoints {d0:.4f}/{d25:.4f} do not match the "
              f"no-MK3 and headline grand means")
    print("\nCHECK: " + ("PASS" if ok else "FAIL"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()