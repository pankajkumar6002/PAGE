"""Per-head Ada-KV analysis: does the headline survive a per-head plain arm?

Evaluates P1 (per budget, as registered), P3 and P4. P2 and P5 are NOT checked
here: P5 gate-invariance is asserted by the runner against the released logs,
and P2 is a magnitude claim reported in the per-task table rather than gated.

Reads the paired runs written by adakv_matrix.py and evaluates every
pre-registered prediction P1-P5 explicitly, including the ones that would
damage the paper. Cells still running are reported as incomplete rather than
partially summarised: a short file once produced an SD of 0.246 against a true
0.01 in the prior round, so partial files are refused, not averaged.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paths import MODELS_HF as MODELS, TAU, MK3, adakv_cell, out_path, verify_prereg

TASKS = ["niah_multikey_3", "vt", "fwe", "qa_1"]
BUDGETS = [0.0625, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
EXPECTED_ROWS = len(TASKS) * 100 * len(BUDGETS)
AGGRESSIVE = 0.25
P3_FLOOR = 0.15


def load(slug):
    path = adakv_cell(slug)
    if not os.path.exists(path):
        return None, "not started"
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if len(rows) != EXPECTED_ROWS:
        kind = "incomplete" if len(rows) < EXPECTED_ROWS else "OVER-LONG (duplicated rows?)"
        return rows, f"{kind} ({len(rows)}/{EXPECTED_ROWS} rows)"
    return rows, None


def deltas(rows, budget_filter=lambda b: True, task_filter=lambda t: True):
    """(delta_shared, delta_adakv, n) over the selected rows."""
    sel = [r for r in rows
           if budget_filter(r["budget"]) and task_filter(r["task"])]
    if not sel:
        return float("nan"), float("nan"), 0
    def d(alloc):
        g = sum(bool(r[f"correct_gated_{alloc}"]) for r in sel)
        p = sum(bool(r[f"correct_plain_{alloc}"]) for r in sel)
        return (g - p) / len(sel)
    return d("shared"), d("adakv"), len(sel)


def main():
    print("prereg (predictions section) sha256:", verify_prereg()[:16], "...")
    cells, incomplete = {}, {}
    for name, slug, _ in MODELS:
        rows, err = load(slug)
        if err:
            incomplete[name] = err
            if not rows:
                continue
        cells[name] = rows

    complete = {k: v for k, v in cells.items() if k not in incomplete}

    lines = [
        "# The headline matrix with a per-head Ada-KV plain arm",
        "",
        "Both plain arms are recorded on the SAME inputs in the SAME run, so",
        "every comparison below is paired. The only thing that varies is the",
        "allocation of a fixed token budget across kv-heads.",
        "",
    ]
    if incomplete:
        lines += ["> **Incomplete cells** (excluded from every aggregate):", ""]
        for name, err in incomplete.items():
            lines.append(f"> - {name}: {err}")
        lines.append("")

    if not complete:
        lines += ["No cell is complete yet. Nothing is reported.", ""]
        path = out_path("adakv_matrix.md")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print("\n".join(lines))
        print("CHECK: PENDING")
        raise SystemExit(2)

    # --- headline table -----------------------------------------------------
    lines += [
        "## Delta (gated minus plain), by allocation",
        "",
        "| cell | budgets | Delta shared | Delta Ada-KV | change | n |",
        "|---|---|---:|---:|---:|---:|",
    ]
    agg = {"all": [], "aggr": []}
    for name, rows in complete.items():
        ds, da, n = deltas(rows)
        lines.append(f"| {name} | all b<1.0 | {ds:+.4f} | {da:+.4f} | {da-ds:+.4f} | {n} |")
        agg["all"].append((ds, da))
        ds2, da2, n2 = deltas(rows, budget_filter=lambda b: b <= AGGRESSIVE)
        lines.append(f"| {name} | b<=0.25 | {ds2:+.4f} | {da2:+.4f} | {da2-ds2:+.4f} | {n2} |")
        agg["aggr"].append((ds2, da2))

    gm_s = sum(x for x, _ in agg["all"]) / len(agg["all"])
    gm_a = sum(y for _, y in agg["all"]) / len(agg["all"])
    gma_s = sum(x for x, _ in agg["aggr"]) / len(agg["aggr"])
    gma_a = sum(y for _, y in agg["aggr"]) / len(agg["aggr"])
    lines += [
        f"| **mean** | all b<1.0 | **{gm_s:+.4f}** | **{gm_a:+.4f}** | **{gm_a-gm_s:+.4f}** | |",
        f"| **mean** | b<=0.25 | **{gma_s:+.4f}** | **{gma_a:+.4f}** | **{gma_a-gma_s:+.4f}** | |",
        "",
    ]

    # --- per-task -----------------------------------------------------------
    lines += ["## Per-task Delta at b <= 0.25", "",
              "| cell | task | Delta shared | Delta Ada-KV | plain shared | plain Ada-KV |",
              "|---|---|---:|---:|---:|---:|"]
    task_order_ok = True
    for name, rows in complete.items():
        per = {}
        for t in TASKS:
            ds, da, n = deltas(rows, lambda b: b <= AGGRESSIVE, lambda x: x == t)
            sel = [r for r in rows if r["budget"] <= AGGRESSIVE and r["task"] == t]
            ps = sum(bool(r["correct_plain_shared"]) for r in sel) / len(sel) if sel else float("nan")
            pa = sum(bool(r["correct_plain_adakv"]) for r in sel) / len(sel) if sel else float("nan")
            per[t] = da
            lines.append(f"| {name} | {t} | {ds:+.4f} | {da:+.4f} | {ps:.3f} | {pa:.3f} |")
        if per.get(MK3, -9) < max(v for k, v in per.items() if k != MK3):
            task_order_ok = False
    lines.append("")

    # --- predictions --------------------------------------------------------
    # P1 is registered PER BUDGET, per cell ("for every budget b, every cell"),
    # not on budget-aggregated deltas. Aggregation can cancel violations, so
    # evaluate it as written.
    p1_viol = []
    for name, rows in complete.items():
        for b in BUDGETS:
            ds, da, n = deltas(rows, budget_filter=lambda x, bb=b: abs(x - bb) < 1e-9)
            if n and da > ds + 1e-9:
                p1_viol.append((name, b, ds, da))
    p1 = not p1_viol
    p1_agg = all(da <= ds + 1e-9 for ds, da in agg["all"])
    p3 = gma_a >= P3_FLOOR
    lines += [
        "## Pre-registered predictions",
        "",
        "| id | prediction | outcome |",
        "|---|---|---|",
        f"| P1 | Ada-KV delta <= shared delta in every cell | "
        f"{'HOLDS' if p1 else '**VIOLATED**'} |",
        f"| P3 | mean Ada-KV delta at b<=0.25 >= {P3_FLOOR:+.2f} | "
        f"{'HOLDS' if p3 else '**VIOLATED**'} (measured {gma_a:+.4f}) |",
        f"| P4 | MK3 keeps the largest per-task delta | "
        f"{'HOLDS' if task_order_ok else '**VIOLATED**'} |",
        "",
        "## Reading",
        "",
    ]
    if p3 and p1:
        lines += [
            f"The headline survives a clean per-head baseline: at the >= 4x",
            f"budgets the caption endorses, the delta is {gma_a:+.4f} against",
            f"{gma_s:+.4f} with the single mask. The 'single-mask artifact'",
            "concession in Limitations can be replaced with a measurement.",
        ]
    else:
        lines += [
            f"**The headline does not survive.** At b <= 0.25 the Ada-KV delta is",
            f"{gma_a:+.4f}, below the pre-registered floor of {P3_FLOOR:+.2f}.",
            "Per the falsification table this is a major finding: the matrix",
            "result is substantially a single-mask artifact. Escalate before",
            "writing.",
        ]
    if not p1:
        lines += [
            "",
            f"**P1 violated at {len(p1_viol)} (cell, budget) points** "
            f"(aggregated over budgets it {'also fails' if not p1_agg else 'appears to hold'}):",
            "",
            "| cell | b | Delta shared | Delta Ada-KV | excess |",
            "|---|---:|---:|---:|---:|",
        ]
        for nm, b, ds, da in p1_viol:
            lines.append(f"| {nm} | {b} | {ds:+.4f} | {da:+.4f} | {da-ds:+.4f} |")
        lines += [
            "",
            "The pre-registration reads P1 failure as a code bug. That call needs",
            "a magnitude check before it is accepted: seed SD measured in the",
            "prior round was 0.004-0.011, so excesses inside that band are noise",
            "rather than evidence of a defect. Audit the implementation against",
            "the reference before either reporting or dismissing these numbers.",
        ]

    ok = p1 and p3 and task_order_ok and bool(complete)
    lines += ["", "## Checks", "",
              f"- [{'PASS' if p1 else 'FAIL'}] P1 per-budget direction (Ada-KV no stronger delta than shared)",
              f"- [{'PASS' if complete else 'FAIL'}] at least one complete cell "
              f"({len(complete)}/{len(MODELS)})",
              f"- [{'PASS' if p3 else 'FAIL'}] P3 survival at b<=0.25: {gma_a:+.4f} vs floor {P3_FLOOR:+.2f}",
              f"- [{'PASS' if task_order_ok else 'FAIL'}] P4 task ordering preserved: {task_order_ok}",
              "- [INFO] P2 and P5 are not evaluated here: P2 needs the b=0.5 MK3",
              "  breakdown and P5 is verified at run time by the runner's guards."]

    path = out_path("adakv_matrix.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {path}")
    print("CHECK: PASS" if ok else "CHECK: FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()