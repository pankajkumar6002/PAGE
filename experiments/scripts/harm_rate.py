"""P0-1 replacement metric: fixed-budget harm rate Pr_x[A(b, x) < A_full(x)].

The recovery indicator r(x) of App. `app:recovery-hist` maximises over the
budget sweep, which includes b = 1.0 where the gated arm reproduces the full
cache (see audit_budget1_identity.py). Its r = -1 bin is therefore empty by
construction and says nothing about harm.

Holding the budget fixed makes "worse than the full cache" a real event. The
gated-vs-plain contrast at the same b is then the quantity a reviewer wants:
how often does the deployed system lose an answer it would otherwise have got?

PROVENANCE NOTE. There are two files named for the 16K Qwen2.5-1.5B cell:

  gated_16k_qwen15b_sdpa.jsonl   1800 rows, 4 tasks, 200 ids   <- the real cell,
                                 the one gated_16k_qwen15b_sdpa_report.md and
                                 the paper report
  gated_16k_qwen15b.jsonl        9 rows, NIAH-MK3 only, id=0   <- an aborted
                                 eager-attention stub

Pooling the stub drops 50 inputs per task from FWE/QA_1/VT (N falls 500 -> 450)
and inflates every harm rate. The revision plan's table did exactly that. This
script reports both pools so the discrepancy stays auditable, and asserts both,
but only the corrected pool is promoted into the paper.
"""
import os
from collections import defaultdict

from paths import RESULTS, TAU, out_path
from gatelib import eviction_rows, rate, read_rows

BASE_CELLS = [
    ("gated_4k_qwen15b.jsonl", "snapkv", "qwen15b"),
    ("gated_4k_qwen3b.jsonl", "snapkv", "qwen3b"),
    ("gated_4k_qwen14b.jsonl", "snapkv", "qwen14b"),
    ("gated_4k_mistral7b.jsonl", "snapkv", "mistral7b"),
    ("gated_16k_qwen3b.jsonl", "snapkv", "qwen3b16k"),
    ("gated_16k_mistral7b.jsonl", "snapkv", "mistral7b16k"),
]
CORRECTED = BASE_CELLS + [("gated_16k_qwen15b_sdpa.jsonl", "snapkv", "qwen15b16k")]
PLAN = BASE_CELLS + [("gated_16k_qwen15b.jsonl", "snapkv", "qwen15b16k")]

TASK_ORDER = ["niah_multikey_3", "fwe", "qa_1", "vt"]
PRETTY = {"niah_multikey_3": "NIAH-MK3", "fwe": "FWE", "qa_1": "QA_1", "vt": "VT"}
REPORT_B = (0.0625, 0.5)

# Recomputed over the corrected 7-cell pool at tau = 0.07. Promoted to the paper.
EXPECTED_CORRECTED = {
    ("niah_multikey_3", 0.0625): (0.026, 0.750),
    ("niah_multikey_3", 0.5): (0.018, 0.546),
    ("fwe", 0.0625): (0.236, 0.414),
    ("fwe", 0.5): (0.064, 0.116),
    ("qa_1", 0.0625): (0.110, 0.110),
    ("qa_1", 0.5): (0.010, 0.010),
    ("vt", 0.0625): (0.684, 0.694),
    ("vt", 0.5): (0.026, 0.026),
}
# The revision plan's table. Asserted so the diagnosis stays reproducible.
EXPECTED_PLAN = {
    ("niah_multikey_3", 0.0625): (0.029, 0.796),
    ("niah_multikey_3", 0.5): (0.020, 0.594),
    ("fwe", 0.0625): (0.291, 0.444),
    ("fwe", 0.5): (0.071, 0.120),
    ("qa_1", 0.0625): (0.118, 0.118),
    ("qa_1", 0.5): (0.009, 0.009),
    ("vt", 0.0625): (0.751, 0.756),
    ("vt", 0.5): (0.022, 0.022),
}


def pool(cells, stored=False):
    """(task, budget) -> harm indicators for the gated and plain arms."""
    harm = defaultdict(lambda: {"gated": [], "plain": []})
    used = []
    for fn, policy, slug in cells:
        path = os.path.join(RESULTS, fn)
        if not os.path.exists(path):
            print(f"MISSING: {fn}")
            continue
        used.append((fn, len(read_rows(path))))
        for r in eviction_rows(path, policy, slug, stored=stored):
            s = harm[(r["task"], r["budget"])]
            s["gated"].append(int(r["gated"] < r["full"]))
            s["plain"].append(int(r["plain"] < r["full"]))
    return harm, used


def table(harm, budgets, emit):
    emit("| task | " + " | ".join(f"b={b:g} gated / plain" for b in budgets) + " |")
    emit("|---" * (len(budgets) + 1) + "|")
    for t in TASK_ORDER:
        cells = []
        for b in budgets:
            s = harm.get((t, b))
            cells.append(f"{rate(s['gated']):.3f} / {rate(s['plain']):.3f}"
                         if s and s["gated"] else "--")
        emit(f"| {PRETTY[t]} | " + " | ".join(cells) + " |")


def assert_pool(harm, expected, label):
    ok = True
    print(f"\n## Assertions: {label}\n")
    for (t, b), (eg, ep) in sorted(expected.items()):
        s = harm.get((t, b))
        if not s:
            print(f"FAIL {PRETTY[t]} b={b}: no data")
            ok = False
            continue
        g, p = rate(s["gated"]), rate(s["plain"])
        good = abs(g - eg) < 1e-3 and abs(p - ep) < 1e-3
        ok &= good
        print(f"{'ok  ' if good else 'FAIL'} {PRETTY[t]:9s} b={b:<7g} "
              f"got {g:.3f}/{p:.3f}  expected {eg:.3f}/{ep:.3f}")
    return ok


def main():
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    harm, used = pool(CORRECTED)
    budgets = sorted({b for _, b in harm})

    emit(f"# Fixed-budget harm rate, tau = {TAU}")
    emit()
    emit("Pr_x[ A(b, x) < A_full(x) ]: the fraction of inputs an arm gets wrong "
         "that the full cache gets right.")
    emit()
    emit(f"## Corrected pool ({len(used)} SnapKV cells) - PROMOTED TO THE PAPER")
    emit()
    table(harm, budgets, emit)
    ns = {k: len(v["gated"]) for k, v in harm.items()}
    emit()
    emit(f"N per (task, budget): min {min(ns.values())}, max {max(ns.values())}")
    emit()
    emit("Cells pooled:")
    for fn, n in used:
        emit(f"- `{fn}` ({n} rows)")

    mk3 = harm[("niah_multikey_3", 0.0625)]
    g, p = rate(mk3["gated"]), rate(mk3["plain"])
    emit()
    emit(f"NIAH-MK3 at b = 0.0625: harm falls {p:.3f} -> {g:.3f}, a {p/g:.0f}x "
         f"reduction. Harm is reduced, not eliminated.")
    inert = [PRETTY[t] for t in TASK_ORDER
             if abs(rate(harm[(t, 0.0625)]["gated"])
                    - rate(harm[(t, 0.0625)]["plain"])) < 1e-9]
    emit(f"Gate inert at b = 0.0625 (gated == plain): {', '.join(inert) or 'none'}")
    vt = harm[("vt", 0.0625)]
    emit(f"VT at b = 0.0625 is harmed either way: {rate(vt['plain']):.3f} plain vs "
         f"{rate(vt['gated']):.3f} gated.")

    # The plan read `correct_gated` off the log rather than re-evaluating at
    # tau=0.07, so its Qwen2.5-3B 4K cell is at tau=0.04. Reproduce both
    # choices together, otherwise the diagnosis is not exact.
    harm_plan, used_plan = pool(PLAN, stored=True)
    emit()
    emit("## The revision plan's pool, for provenance")
    emit()
    stub = [n for fn, n in used_plan if fn == "gated_16k_qwen15b.jsonl"]
    emit("Two differences from the corrected pool, both of which move the "
         "numbers:")
    emit()
    emit(f"1. It substitutes `gated_16k_qwen15b.jsonl` "
         f"({stub[0] if stub else '?'} rows, NIAH-MK3 only, one input) for the "
         f"1800-row `gated_16k_qwen15b_sdpa.jsonl`, so the 16K Qwen2.5-1.5B "
         f"cell is absent from FWE, QA_1 and VT and N falls to about 450.")
    emit("2. It reads `correct_gated` off the log instead of re-evaluating at "
         "tau = 0.07, so the Qwen2.5-3B 4K cell enters at its executed "
         "tau = 0.04. This lowers the gated harm rate on FWE and VT only.")
    emit()
    table(harm_plan, REPORT_B, emit)
    ns_plan = {k: len(v["gated"]) for k, v in harm_plan.items()}
    emit()
    emit(f"N per (task, budget): min {min(ns_plan.values())}, "
         f"max {max(ns_plan.values())}")

    with open(out_path("harm_rate.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    ok = assert_pool(harm, EXPECTED_CORRECTED, "corrected pool (promoted)")
    ok &= assert_pool(harm_plan, EXPECTED_PLAN,
                      "revision plan's pool (diagnosis must stay reproducible)")
    print("\nCHECK: " + ("PASS" if ok else "FAIL"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()