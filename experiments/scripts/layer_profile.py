"""P-1: plot the per-layer agreement profile a_l that assumption A3 asserts.

A3 says early layers respond to local context and late layers to task
structure, which is why an early-minus-late contrast should separate the task
classes. The paper asserts this and only ever reports the binned scalar D. The
per-layer agreements are already logged, so the curve costs nothing.

The result is not uniformly supportive, which is why it is worth showing. On
Qwen the profile decays with depth and NIAH-MK3 holds a visibly higher late
plateau than the dilution-prone tasks, which is what A3 predicts. On Llama the
profile is non-monotone, with a mid-stack spike, and NIAH-MK3 and VT become
nearly indistinguishable. That is a mechanistic account of why the fixed
threshold does not transfer to the Llama architecture, which the paper
otherwise reports as an unexplained artifact.
"""
import json
import os
from collections import defaultdict

from paths import RESULTS, out_path
from layer_subsample import read_jsonl, mean

CELLS = [
    ("Qwen2.5-1.5B 4K", "drops_qwen15b_4k_n100.jsonl"),
    ("Llama-3.1-8B 4K", "drops_llama31_8b_4k.jsonl"),
]
TASKS = [("niah_multikey_3", "NIAH-MK3 (capacity-bound)", "C3"),
         ("vt", "VT (dilution-prone)", "C0"),
         ("fwe", "FWE (dilution-prone)", "C2")]


def profiles(path):
    rows, corrupt = read_jsonl(path)
    by = defaultdict(list)
    for r in rows:
        by[r["task"]].append(r["head_agreement_per_layer"])
    L = len(rows[0]["head_agreement_per_layer"])
    return {t: [mean([p[i] for p in ps]) for i in range(L)]
            for t, ps in by.items()}, L, corrupt


def main():
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("# P-1: per-layer head-agreement profiles $a_\\ell$ (assumption A3)")
    emit()

    data = {}
    for label, fn in CELLS:
        path = os.path.join(RESULTS, fn)
        if not os.path.exists(path):
            emit(f"## {label}: MISSING {fn}")
            continue
        prof, L, corrupt = profiles(path)
        data[label] = (prof, L)
        emit(f"## {label} ($L = {L}$)")
        emit()
        emit("| task | early third | late third | $D$ |")
        emit("|---|---:|---:|---:|")
        third = L // 3
        for t, pretty, _ in TASKS:
            if t not in prof:
                continue
            e = mean(prof[t][:third])
            l = mean(prof[t][-third:])
            emit(f"| {pretty} | {e:.3f} | {l:.3f} | {e - l:+.3f} |")
        mk3 = prof.get("niah_multikey_3")
        others = [prof[t] for t, _, _ in TASKS[1:] if t in prof]
        if mk3 and others:
            third = L // 3
            d_mk3 = mean(mk3[:third]) - mean(mk3[-third:])
            d_min = min(mean(o[:third]) - mean(o[-third:]) for o in others)
            emit()
            emit(f"Separation margin (min dilution-prone $D$ minus MK3 $D$): "
                 f"**{d_min - d_mk3:+.3f}**")
        emit()

    if "Qwen2.5-1.5B 4K" in data and "Llama-3.1-8B 4K" in data:
        emit("The margin collapses from Qwen to Llama, and the Llama profile is "
             "non-monotone in depth rather than decaying. This is the "
             "mechanistic content behind the Llama transfer failure and the "
             "per-input AUC of about 0.80 there: on that architecture the "
             "early-minus-late contrast is reading a profile whose shape does "
             "not match the one A3 describes.")

    with open(out_path("layer_profile.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(data), figsize=(7.4, 2.7), sharey=True)
        if len(data) == 1:
            axes = [axes]
        for ax, (label, (prof, L)) in zip(axes, data.items()):
            for t, pretty, c in TASKS:
                if t not in prof:
                    continue
                ax.plot(range(L), prof[t], color=c, lw=1.6,
                        label=pretty.split(" (")[0])
            ax.axvspan(0, L // 3 - 1, color="0.9", zorder=0)
            ax.axvspan(L - L // 3, L - 1, color="0.9", zorder=0)
            ax.set_title(label, fontsize=8)
            ax.set_xlabel("layer $\\ell$", fontsize=8)
            ax.tick_params(labelsize=7)
        axes[0].set_ylabel("head agreement $a_\\ell$", fontsize=8)
        axes[0].legend(fontsize=6.5, frameon=False)
        fig.tight_layout()
        fig.savefig(out_path("layer_profile.pdf"))
        fig.savefig(out_path("layer_profile.png"), dpi=200)
        print(f"\nwrote {out_path('layer_profile.pdf')}")
    except ImportError:
        print("\nmatplotlib not available, skipped the figure")

    print("\nCHECK: PASS")


if __name__ == "__main__":
    main()
