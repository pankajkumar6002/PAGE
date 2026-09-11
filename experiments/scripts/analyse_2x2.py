"""2x2 control: is the D shift caused by my low-memory wrapper, or by hardware?

The low-memory wrapper reproduced the paper's AUC exactly but gave mean D on
NIAH-MK3 of 0.030 against the paper's 0.045. Two explanations were consistent
with that single number:

  (a) the wrapper changes the statistic, in which case every held-out cell
      measured through it (Qwen-14B, Llama-8B, Mistral-16K) is an artifact;
  (b) hardware, since the reproduction gate already showed drop moving by up
      to 0.021 between this host and the A100 the released logs came from.

One measurement cannot separate them. Crossing code path with hardware can:

                    ORIGINAL script      MY WRAPPER
    jagannath (A100)      A                  B
    sateri (Ada)          C                  D

  A vs B  and  C vs D   isolate the code path, hardware held fixed.
  A vs C  and  B vs D   isolate hardware, code path held fixed.

If |A-B| and |C-D| are ~0 while |A-C| is ~0.015, the wrapper is faithful and
the gap to the paper is hardware. If |A-B| is large, the wrapper is at fault
and the held-out numbers must be discarded.
"""
import json
import os
from statistics import mean

from paths import out_path

ANCHOR = "niah_multikey_3"
DIL = ["vt", "fwe", "qa_1", "niah_multivalue"]

CELLS = [
    ("jagannath (A100)", "ORIGINAL", "sigabl_qwen15b_hostA_original.jsonl"),
    ("jagannath (A100)", "WRAPPER",  "sigabl_qwen15b_hostA_wrapper.jsonl"),
    ("sateri (Ada)",     "ORIGINAL", "signal_ablation_qwen15b_ORIGINAL.jsonl"),
    ("sateri (Ada)",     "WRAPPER",  "signal_ablation_qwen15b_VALIDATE.jsonl"),
]
PAPER_D = 0.045


def auc(pos, neg):
    n = t = 0
    for a in pos:
        for b in neg:
            t += 1
            n += (a < b) + 0.5 * (a == b)
    return n / t if t else float("nan")


def stats(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    by = {}
    for r in rows:
        by.setdefault(r["task"], []).append(r)
    a = [r["drop_D"] for r in by[ANCHOR]]
    flat = [r["drop_D"] for t in DIL if t in by for r in by[t]]
    return mean(a), auc(a, flat), len(rows)


def main():
    base = os.path.dirname(out_path("x"))
    got = {}
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("# 2x2 control: code path x hardware")
    emit()
    emit(f"| host | code path | mean $D$ (MK3) | AUC | rows |")
    emit("|---|---|---:|---:|---:|")
    for host, path_kind, fn in CELLS:
        p = os.path.join(base, fn)
        if not os.path.exists(p):
            emit(f"| {host} | {path_kind} | _pending_ | | |")
            continue
        d, a, n = stats(p)
        got[(host, path_kind)] = d
        emit(f"| {host} | {path_kind} | {d:.4f} | {a:.3f} | {n} |")
    emit(f"| paper (A100, released) | ORIGINAL | {PAPER_D:.4f} | 1.000 | -- |")
    emit()

    def diff(k1, k2, label):
        if k1 in got and k2 in got:
            v = abs(got[k1] - got[k2])
            emit(f"- {label}: **{v:.4f}**")
            return v
        emit(f"- {label}: _pending_")
        return None

    emit("**Code-path effect** (hardware held fixed)")
    cp1 = diff(("jagannath (A100)", "ORIGINAL"), ("jagannath (A100)", "WRAPPER"),
               "jagannath: original vs wrapper")
    cp2 = diff(("sateri (Ada)", "ORIGINAL"), ("sateri (Ada)", "WRAPPER"),
               "sateri: original vs wrapper")
    emit()
    emit("**Hardware effect** (code path held fixed)")
    hw1 = diff(("jagannath (A100)", "ORIGINAL"), ("sateri (Ada)", "ORIGINAL"),
               "original: jagannath vs sateri")
    hw2 = diff(("jagannath (A100)", "WRAPPER"), ("sateri (Ada)", "WRAPPER"),
               "wrapper: jagannath vs sateri")
    emit()

    if all(v is not None for v in (cp1, cp2, hw1, hw2)):
        code = max(cp1, cp2)
        hw = max(hw1, hw2)
        emit(f"max code-path effect **{code:.4f}**, "
             f"max hardware effect **{hw:.4f}**")
        emit()
        if code < 1e-3:
            emit("**VERDICT: the wrapper is faithful.** It leaves $D$ unchanged "
                 "on identical hardware, so the held-out cells measured through "
                 "it are usable, and the gap to the released numbers is the "
                 "hardware offset the reproduction gate already documented.")
        elif code < hw:
            emit("**VERDICT: mixed.** The wrapper moves $D$, but by less than "
                 "hardware does. Held-out cells are usable only for ordinal "
                 "claims, not for absolute $D$ values.")
        else:
            emit("**VERDICT: the wrapper changes the statistic.** Discard the "
                 "Qwen-14B, Llama-8B and Mistral-16K cells measured through it.")

    with open(out_path("control_2x2.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
