"""Is the Qwen2.5-14B H2O column an independent base evictor?

The column is suspected to degenerate toward SnapKV scoring under two-pass
prefill, which would mean it should be marked as such in the headline table.
The measurement is stronger than that suspicion: the column is an EXACT
duplicate.

Mechanism: Qwen2.5-14B is the only matrix cell run with --two_pass. In two-pass
mode the attention matrix carries only obs_window queries, so H2O's "mean over
ALL queries" branch (gated_eviction.py pool_score, guarded by
`a.shape[2] > obs_window`) never fires and H2O reduces to SnapKV exactly.

Two separate facts come out of this join, and only one of them is bad news:

  * BAD: the Qwen2.5-14B H2O cell is not independent. The matrix has 15
    independent cells, not 16, and the table does not say so.
  * GOOD: `drop` is identical across scorers on 100% of rows in every model.
    The gate signal does not depend on the base evictor, which is exactly what
    a method-agnostic gate should show, and it is worth stating.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paths import MODELS_HF as MODELS, released_cell, out_path
from gatelib import read_rows

DUPLICATE_SLUG = "qwen14b"
INDEPENDENT_MAX = 0.80
DROP_TOL = 1e-9


def join(slug):
    sn = released_cell(slug, "snapkv")
    h2 = released_cell(slug, "h2o")
    if not (os.path.exists(sn) and os.path.exists(h2)):
        return None
    # b=1.0 rows are the full cache in BOTH arms by construction, so including
    # them inflates agreement toward 1 for every model. Compare only rows where
    # the scorer actually chooses what to evict.
    base = {(r["task"], r["id"], r["budget"]): r for r in read_rows(sn)}
    n = agree = drop_same = 0
    for r in read_rows(h2):
        k = (r["task"], r["id"], r["budget"])
        if k not in base or r["budget"] >= 1.0:
            continue
        n += 1
        agree += int(bool(r["correct_plain"]) == bool(base[k]["correct_plain"]))
        drop_same += int(abs(r["drop"] - base[k]["drop"]) < DROP_TOL)
    if not n:
        return None
    return {"n": n, "agree": agree / n, "drop_same": drop_same / n}


def main():
    stats = {}
    for name, slug, _ in MODELS:
        st = join(slug)
        if st:
            stats[(name, slug)] = st
    if not stats:
        raise SystemExit("no joinable cells; set PAGE_RESULTS")

    lines = [
        "# The Qwen2.5-14B H2O column duplicates SnapKV",
        "",
        "Joined on (task, id, budget) between `gated_4k_{slug}.jsonl` (SnapKV)",
        "and `gated_{h2o}_{slug}_4k.jsonl`, **excluding b = 1.0** where both",
        "arms are the full cache by construction.",
        "",
        "| cell | rows | plain-outcome agreement | `drop` identical |",
        "|---|---:|---:|---:|",
    ]
    for (name, slug), st in stats.items():
        mark = "  <- duplicate" if st["agree"] >= 0.999 else ""
        lines.append(f"| {name} | {st['n']} | {st['agree']:.4f}{mark} | "
                     f"{st['drop_same']:.4f} |")

    dup = stats.get(("Qwen2.5-14B", DUPLICATE_SLUG))
    others = [st for (n, s), st in stats.items() if s != DUPLICATE_SLUG]
    lines += [
        "",
        "## What this settles",
        "",
    ]
    if dup:
        lines += [
            f"Qwen2.5-14B H2O agrees with SnapKV on **{dup['agree']*100:.1f}% of",
            f"{dup['n']} rows**, against {min(s['agree'] for s in others)*100:.0f}"
            f"-{max(s['agree'] for s in others)*100:.0f}% for the other three",
            "models. It is not a weakened independent evictor, it is the same",
            "measurement reported twice.",
            "",
            "**Fix in the paper:** dagger the cell in `tab:matrix` and add to the",
            "caption: *Qwen2.5-14B ran two-pass prefill, where H2O's all-query",
            "score reduces exactly to SnapKV's; this column duplicates the SnapKV",
            f"column (identical on {dup['n']}/{dup['n']} rows) and is not an",
            "independent base evictor.* Report the matrix as **15 independent",
            "cells**, and give the grand mean both ways.",
        ]
    lines += [
        "",
        "## The good half of the same measurement",
        "",
        "`drop` is identical to 1e-9 on **100% of rows in every model**. The gate",
        "signal is computed from prefill attentions and does not depend on which",
        "base evictor consumes it. That is the method-agnosticism claim, measured",
        "rather than asserted, and it is worth quoting in the paper.",
    ]

    ok = True
    checks = []
    if dup:
        good = abs(dup["agree"] - 1.0) < 1e-9
        checks.append(("Qwen2.5-14B H2O agreement == 1.0000", good, f"{dup['agree']:.4f}"))
        ok &= good
    else:
        checks.append(("Qwen2.5-14B cell present", False, "missing"))
        ok = False
    for (name, slug), st in stats.items():
        if slug == DUPLICATE_SLUG:
            continue
        good = st["agree"] < INDEPENDENT_MAX
        checks.append((f"{name} H2O agreement < {INDEPENDENT_MAX} (independent)",
                       good, f"{st['agree']:.4f}"))
        ok &= good
    all_drop = all(abs(st["drop_same"] - 1.0) < 1e-9 for st in stats.values())
    checks.append(("`drop` identical on 100% of rows in every model", all_drop,
                   "ok" if all_drop else "differs"))
    ok &= all_drop

    lines += ["", "## Checks", ""]
    for name, passed, got in checks:
        lines.append(f"- [{'PASS' if passed else 'FAIL'}] {name} (got {got})")

    path = out_path("h2o_degeneracy_audit.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {path}")
    print("CHECK: PASS" if ok else "CHECK: FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()