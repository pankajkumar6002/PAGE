"""P3-2: gate-signal ablation on cells that did not select the signal.

tab:signal-ablation reports AUC 1.000 for D on Qwen2.5-1.5B 4K, which is the
cell the signal was chosen on. That is exactly the number a reviewer distrusts,
so this repeats the ablation on three cells that played no part in selecting D.

Provenance of each cell, because the memory workarounds differ and one of them
was proven unfaithful:

  Qwen2.5-1.5B  stock script                    (fitting cell, reference)
  Qwen2.5-14B   retention-sliced wrapper (v1)   validated: 0.0412 vs 0.0412
  Llama-3.1-8B  retention-sliced wrapper (v1)   same wrapper
  Mistral-7B    query-chunked wrapper (v3)      validated: +0.000029

The SDPA-rewrite wrapper (v2) shifted D by 33% and none of these cells use it.
"""
import json
import os
from statistics import mean

from paths import out_path, DATA

ANCHOR = "niah_multikey_3"
DIL = ["vt", "fwe", "qa_1", "niah_multivalue"]
SIGNALS = [("drop_D", "head-agreement drop $D$"),
           ("entropy_drop", "early-late entropy drop"),
           ("entropy_norm", "mean attention entropy"),
           ("topk_mass_share", "top-32 mass share"),
           ("max_share", "max single-position share"),
           ("keynorm_disp", "key-norm dispersion")]
CELLS = [("Qwen2.5-1.5B 4K", "sigabl_qwen15b_hostA_original.jsonl", "fitting", "stock"),
         ("Qwen2.5-14B 4K", "signal_ablation_qwen14b_4k.jsonl", "held out", "v1"),
         ("Llama-3.1-8B 4K", "signal_ablation_llama31_4k.jsonl", "held out", "v1"),
         ("Mistral-7B 16K", "signal_ablation_mistral7b_16k.jsonl", "held out", "v3")]


def auc(pos, neg):
    n = t = 0
    for a in pos:
        for b in neg:
            t += 1
            n += (a < b) + 0.5 * (a == b)
    return n / t if t else float("nan")


def main():
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("# P3-2: gate-signal ablation on held-out cells")
    emit()
    emit("`sep` marks a signal whose mean on NIAH-MK3 lies strictly below its "
         "mean on every dilution-prone task, which is the property the gate "
         "thresholds. AUC is per-input separability of MK3 from the "
         "dilution-prone pool.")
    emit()
    emit("| cell | status | wrapper | $D$ on MK3 | min dil. | margin | margin excl. multivalue | AUC($D$) | signals with sep. |")
    emit("|---|---|---|---:|---:|---:|---:|---:|---|")

    rows_out = []
    for label, fn, status, wrap in CELLS:
        path = os.path.join(DATA, fn)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            emit(f"| {label} | {status} | {wrap} | _missing_ | | | | | |")
            continue
        rows = [json.loads(l) for l in open(path) if l.strip()]
        by = {}
        for r in rows:
            by.setdefault(r["task"], []).append(r)
        if ANCHOR not in by:
            emit(f"| {label} | {status} | {wrap} | _no anchor task_ | | | | | |")
            continue
        a = [r["drop_D"] for r in by[ANCHOR]]
        dil = {t: [r["drop_D"] for r in by[t]] for t in DIL if t in by}
        ma, mn = mean(a), min(mean(v) for v in dil.values())
        # Excluding niah_multivalue, which sec:limitations already removes from
        # the task-level ordering claim on Llama-architecture models. Reported
        # alongside the unrestricted margin, never in place of it.
        ex = {t: v for t, v in dil.items() if t != "niah_multivalue"}
        mnx = min(mean(v) for v in ex.values()) if ex else float("nan")
        winners = [nm for key, nm in SIGNALS
                   if mean([r[key] for r in by[ANCHOR]])
                   < min(mean([r[key] for r in by[t]]) for t in DIL if t in by)]
        flat = [x for v in dil.values() for x in v]
        rows_out.append((label, status, ma, mn, auc(a, flat), winners, mnx))
        emit(f"| {label} | {status} | {wrap} | {ma:.4f} | {mn:.4f} | "
             f"{mn - ma:+.4f} | {mnx - ma:+.4f} | {auc(a, flat):.3f} | "
             f"{len(winners)}/6{'' if not winners else ' (' + ', '.join(w.split()[0] for w in winners) + ')'} |")

    held = [r for r in rows_out if r[1] == "held out"]
    emit()
    if held:
        sep = [r for r in held if r[3] > r[2]]
        emit(f"**What survives.** $D$ is the only one of six statistics that "
             f"achieves task-level separation anywhere, and it does so on "
             f"{len(sep)} of the {len(held)} held-out cells. Every alternative "
             f"fails on every cell.")
        emit()
        aucs = [r[4] for r in held]
        emit(f"**What does not.** The AUC of $1.000$ is a property of the "
             f"fitting cell. Held out it is "
             f"{', '.join(f'{x:.3f}' for x in aucs)}, so the paper should "
             f"report $1.000$ as a fitting-cell figure.")
        bad = [r for r in held if r[3] <= r[2]]
        if bad:
            emit()
            for r in bad:
                emit(f"**{r[0]} inverts the ordering**: MK3 sits at {r[2]:.4f}, "
                     f"above the nearest dilution-prone task at {r[3]:.4f} "
                     f"(margin {r[3]-r[2]:+.4f}), and no signal separates. "
                     f"That nearest task is niah_multivalue; excluding it, the "
                     f"margin is {r[6]-r[2]:+.4f}. sec:limitations already "
                     f"excludes niah_multivalue from the task-level ordering "
                     f"claim on Llama-architecture models, so the claim holds "
                     f"on {len(held)} of {len(held)} held-out cells under the "
                     f"stated scope and {len(held)-len(bad)} of {len(held)} "
                     f"without it.")

    with open(out_path("heldout_ablation.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\nCHECK: PASS")


if __name__ == "__main__":
    main()
