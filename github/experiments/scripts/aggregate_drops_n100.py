"""Aggregate per-layer-agreement jsonl files into the per-task mean head-agreement
drop D = a_early - a_late, with stderr and N per task.

Reads any number of (cell_label, jsonl_path) and emits:
- one markdown table per cell with mean D, std, stderr
- a combined table matching tab:drops layout (tasks rows x cells columns)
"""
import argparse
import json
import math
from collections import defaultdict


TASK_ORDER = ["qa_1", "qa_2", "vt", "niah_multivalue", "fwe", "niah_multikey_3"]
TASK_PRETTY = {
    "qa_1": "qa\\_1",
    "qa_2": "qa\\_2",
    "vt": "vt",
    "niah_multivalue": "niah\\_multivalue",
    "fwe": "fwe",
    "niah_multikey_3": "niah\\_mk\\_3",
}


def load_drops(path):
    """Return dict task -> list of D values."""
    by_task = defaultdict(list)
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            pl = r["head_agreement_per_layer"]
            L = len(pl)
            early = pl[: L // 3]
            late = pl[2 * L // 3 :]
            D = sum(early) / len(early) - sum(late) / len(late)
            by_task[r["task"]].append(D)
    return by_task


def stats(vals):
    n = len(vals)
    if n == 0:
        return None
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / n
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    return n, m, sd, se


def fmt(x, sig=3):
    if x is None:
        return "--"
    if x < 0:
        return f"$-{abs(x):.{sig}f}$"
    return f"{x:.{sig}f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cell", action="append", default=[],
                   help="LABEL=PATH, repeatable")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    cells = []
    for spec in args.cell:
        label, path = spec.split("=", 1)
        cells.append((label, path, load_drops(path)))

    out_lines = []
    out_lines.append("# Head-agreement drop D, N=100 per task per cell")
    out_lines.append("")
    out_lines.append("D = a_early - a_late, where a_early is the mean per-layer head-agreement (Jaccard top-32 across head pairs) over the first L/3 layers, and a_late is the mean over the last L/3 layers. One scalar D per input. Reported below: mean D, stderr (sd / sqrt(N)), N.")
    out_lines.append("")

    # Per-cell long-form summary
    for label, path, by_task in cells:
        out_lines.append(f"## {label}")
        out_lines.append(f"Source: `{path}`")
        out_lines.append("")
        out_lines.append("| task | N | mean D | std | stderr |")
        out_lines.append("|---|---:|---:|---:|---:|")
        for t in TASK_ORDER:
            if t not in by_task:
                out_lines.append(f"| {t} | -- | -- | -- | -- |")
                continue
            n, m, sd, se = stats(by_task[t])
            s = "-" if m >= 0 else ""
            out_lines.append(f"| {t} | {n} | {m:+.4f} | {sd:.4f} | {se:.4f} |")
        out_lines.append("")

    # tab:drops style combined table
    out_lines.append("## tab:drops (N=100) — paper-ready replacement")
    out_lines.append("")
    header = "Task & " + " & ".join(label for label, _, _ in cells) + r" \\"
    out_lines.append("```latex")
    out_lines.append(r"\begin{table}[h]")
    out_lines.append(r"\centering")
    out_lines.append(r"\caption{Per-task mean head-agreement drop $D$ on Qwen, Mistral, and Yi-1.5 cells, $N=100$ inputs per task (Yi-1.5-9B $N=50$). Standard error in parentheses. NIAH-MK3 is the smallest drop in five of six architecture columns; on Yi-1.5 it ranks 2nd-smallest behind niah\\_multivalue (model-specific anomaly).}")
    out_lines.append(r"\label{tab:drops}")
    out_lines.append(r"\begin{tabular}{l" + "c" * len(cells) + "}")
    out_lines.append(r"\toprule")
    out_lines.append(header)
    out_lines.append(r"\midrule")
    for t in TASK_ORDER:
        row = [TASK_PRETTY[t]]
        for label, _, by_task in cells:
            if t not in by_task or not by_task[t]:
                row.append("--")
                continue
            n, m, sd, se = stats(by_task[t])
            # format mean with stderr in parens
            m_str = f"${m:+.3f}$".replace("+", "")
            row.append(f"{m_str} ({se:.3f})")
        out_lines.append(" & ".join(row) + r" \\")
    out_lines.append(r"\bottomrule")
    out_lines.append(r"\end{tabular}")
    out_lines.append(r"\end{table}")
    out_lines.append("```")
    out_lines.append("")

    # Stderr headline reduction vs N=20 baseline (since old table was N=20-50)
    out_lines.append("## Standard-error headline")
    out_lines.append("")
    out_lines.append("| cell | task | N | stderr |")
    out_lines.append("|---|---|---:|---:|")
    for label, _, by_task in cells:
        for t in TASK_ORDER:
            if t not in by_task:
                continue
            n, _, _, se = stats(by_task[t])
            out_lines.append(f"| {label} | {t} | {n} | {se:.4f} |")

    with open(args.out, "w") as f:
        f.write("\n".join(out_lines))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
