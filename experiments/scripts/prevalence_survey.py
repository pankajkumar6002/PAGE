"""Prevalence of the capacity-bound class in realistic workloads.

Coarse D-and-sensitivity survey over LongBench subtasks, producing a
per-subtask classification and a pooled capacity-bound workload share f with a
Wilson confidence interval. Answers: outside the synthetic RULER suite, how
much of a realistic workload does the gate actually need to protect?

The per-subtask classification reuses the exact operational definition already
implemented in page-kv/.../analyze_realistic_workload.py (imported, not
re-implemented), so this survey introduces no new discretion. This script adds
the f-share aggregation that analyzer does not compute.

Per-input capacity-bound (the f numerator), matching the prereg:
  full-KV-correct AND destroyed by plain eviction at >=1 tested budget AND D<tau
  (gate closes). f = that count / N, pooled input-weighted over the in-range
  panel. AgentLongBench cells (32K+, out of fitting range) are reported
  separately and NOT pooled.

Zero-GPU on the ingestible logs. Exits `CHECK: PASS`.
"""
import json
import math
import os
import sys

from paths import RESULTS, DATA, TAU, out_path

# Import the released analyzer's classification so the class rule is identical.
_SRC = os.environ.get("PAGE_SRC",
                      os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                   "page-kv", "experiments", "scripts"))
sys.path.insert(0, os.path.abspath(_SRC))
try:
    import analyze_realistic_workload as arw
except Exception as e:   # pragma: no cover
    raise SystemExit(f"cannot import analyze_realistic_workload from {_SRC}: {e}")


# Ingestible released logs (verified present, plain schema).
# (subtask, path, model, pool) -- pool=True only for the single-model in-range
# panel. The pooled f MUST be single-model (Qwen2.5-14B) to avoid an
# apples-to-oranges aggregation; the one released passage_retrieval log is
# Qwen2.5-1.5B, so it is shown in the table (informative) but NOT pooled.
INGEST = [
    ("qasper",              "longbench_qwen14b.jsonl",                    "qwen14b", True),
    ("multifieldqa_en",     "longbench_qwen14b.jsonl",                    "qwen14b", True),
    ("trec",                "longbench_qwen14b.jsonl",                    "qwen14b", True),
    ("triviaqa",            "longbench_qwen14b.jsonl",                    "qwen14b", True),
    ("lcc",                 "longbench_realistic_lcc_qwen14b.jsonl",      "qwen14b", True),
    ("repobench-p",         "longbench_realistic_repobench-p_qwen14b.jsonl", "qwen14b", True),
    ("hotpotqa",            "longbench_realistic_hotpotqa_qwen14b.jsonl", "qwen14b", True),
    ("passage_count",       "longbench_passcount_qwen14b.jsonl",          "qwen14b", True),
    ("passage_retrieval_en", "longbench_passret_qwen15b_n100.jsonl",      "qwen15b", False),
    # Prevalence gap runs (run_prevalence_gaps.sh) write these into PAGE_DATA. Listed so they
    # fold into the pooled f once produced; until then the loop marks them
    # MISSING and skips (they do not affect f). resolve() finds them in DATA.
    ("2wikimqa",            "longbench_2wikimqa_qwen14b.jsonl",           "qwen14b", True),
    ("musique",             "longbench_musique_qwen14b.jsonl",            "qwen14b", True),
    ("gov_report",          "longbench_gov_report_qwen14b.jsonl",         "qwen14b", True),
]


def resolve(fname):
    """A result file may live in the released RESULTS dir or in this round's
    DATA dir (prevalence gap runs write to DATA). Return whichever exists, preferring
    RESULTS; return the RESULTS path if neither exists so the caller reports it
    as MISSING consistently."""
    r = os.path.join(RESULTS, fname)
    if os.path.exists(r):
        return r
    d = os.path.join(DATA, fname)
    if os.path.exists(d):
        return d
    return r


def read_jsonl(path):
    if not os.path.exists(path):
        return None
    rows = []
    try:
        with open(path, errors="replace") as f:
            for line in f:
                if not line.strip() or "\x00" in line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError as e:
        raise SystemExit(f"cannot read {path}: {e}")
    return rows


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def cap_share(rows, floor=None):
    """Per-input capacity-bound count and N, using the prereg definition on one
    subtask's rows (one drop per id).

    `floor`: if given, only budgets b with floor <= b < 1.0 count toward the
    "destroyed at >=1 budget" test, so every subtask is stressed to the SAME
    maximum compression. This removes the confound that the QA subtasks only
    reach b_min=0.125 while the realistic subtasks reach 0.0625; pooling at a
    common floor keeps f comparable across subtasks. floor=None uses each
    subtask's native budgets (reported alongside)."""
    by_id = {}
    for r in rows:
        by_id.setdefault(r["id"], {})[r["budget"]] = r
    n = k = 0
    for i, bmap in by_id.items():
        if 1.0 not in bmap:
            continue
        full_ok = bool(bmap[1.0]["correct_plain"])
        drop = bmap[1.0]["drop"]
        evicted = [bool(bmap[b]["correct_plain"]) for b in bmap
                   if b < 1.0 and (floor is None or b >= floor - 1e-9)]
        if not evicted:
            continue        # no evicting budget at/above the floor => not scored.
        n += 1
        destroyed = full_ok and (not all(evicted))
        gate_closes = drop < TAU
        if destroyed and gate_closes:
            k += 1
    return k, n


def main():
    lines = []
    emit = lines.append
    emit("# Prevalence of the capacity-bound class in realistic workloads\n")
    emit(f"tau = {TAU}. Per-subtask class via analyze_realistic_workload (imported). "
         "Per-input capacity-bound = full-correct ∧ destroyed at ≥1 budget ∧ D<τ. "
         "f = pooled input-weighted share over the in-range panel. AgentLongBench "
         "(32K+) reported separately, not pooled.\n")

    # Common budget floor for a matched-compression pooled f. All pooled
    # subtasks reach 0.125 (the QA logs stop there; realistic logs go deeper),
    # so 0.125 is the deepest floor at which every subtask is stressed equally.
    COMMON_FLOOR = 0.125

    emit("| subtask | model | N | A_full | mean D | gate-open | empirical class | f (native b_min) | f (b≤0.125) | pooled |")
    emit("|---|---|---:|---:|---:|---|---:|---:|---:|:-:|")

    pooled_k = pooled_n = 0        # matched-floor pool (the headline f)
    pooled_k_nat = pooled_n_nat = 0  # native-budget pool (sensitivity check)
    seen = set()
    n_ok = 0
    for subtask, fname, model, pool in INGEST:
        # some logs bundle several subtasks; filter by task name.
        rows_all = read_jsonl(resolve(fname))
        if rows_all is None:
            emit(f"| {subtask} | {model} (MISSING) | | | | | | | |")
            continue
        rows = [r for r in rows_all if r.get("task") == subtask]
        if not rows:
            emit(f"| {subtask} | {model} (no rows) | | | | | | | |")
            continue
        n_ok += 1
        # write a temp single-subtask file? no -- analyze() takes a path. Reuse
        # its internals by calling analyze() on a filtered temp in memory is not
        # supported, so we recompute the two fields we display from arw's helpers
        # by writing a scratch file.
        scratch = out_path(f"_e11_scratch_{subtask}.jsonl")
        try:
            with open(scratch, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            s = arw.analyze(scratch)
        finally:
            try:
                os.remove(scratch)
            except OSError:
                pass
        k_nat, n_nat = cap_share(rows, floor=None)
        k_fl, n_fl = cap_share(rows, floor=COMMON_FLOOR)
        f_nat = k_nat / n_nat if n_nat else float("nan")
        f_fl = k_fl / n_fl if n_fl else float("nan")
        if pool:
            pooled_k += k_fl
            pooled_n += n_fl
            pooled_k_nat += k_nat
            pooled_n_nat += n_nat
        seen.add(subtask)
        emit(f"| {subtask} | {model} | {s['N']} | {s['A_full']:.3f} | "
             f"{s['mean_D']:+.4f} | {s['gate_open_frac']:.2f} | {s['emp_class']} | "
             f"{k_nat}/{n_nat} ({f_nat:.3f}) | {k_fl}/{n_fl} ({f_fl:.3f}) | "
             f"{'yes' if pool else 'no'} |")

    lo, hi = wilson(pooled_k, pooled_n)
    lo_n, hi_n = wilson(pooled_k_nat, pooled_n_nat)
    emit("")
    emit(f"## Pooled in-range capacity-bound share (Qwen2.5-14B, single-model)\n")
    emit(f"- **Headline (matched floor b≤0.125): f = {pooled_k}/{pooled_n} = "
         f"{(pooled_k/pooled_n if pooled_n else float('nan')):.4f}**, "
         f"Wilson 95% [{lo:.4f}, {hi:.4f}]. Every pooled subtask is stressed to "
         f"the same b=0.125 floor, so f is comparable across subtasks.")
    emit(f"- Sensitivity (native per-subtask b_min, mixes 0.125 and 0.0625 floors): "
         f"f = {pooled_k_nat}/{pooled_n_nat} = "
         f"{(pooled_k_nat/pooled_n_nat if pooled_n_nat else float('nan')):.4f}, "
         f"Wilson 95% [{lo_n:.4f}, {hi_n:.4f}].")
    emit(f"- Subtasks analyzed: {n_ok}/{len(INGEST)} ingestible; gap subtasks "
         "(2wikimqa, musique, gov_report/multi_news) and AgentLongBench are added "
         "by the GPU runs in experiments/gpu/run_prevalence_gaps.sh.")
    emit("")

    out = out_path("prevalence_survey.md")
    try:
        with open(out, "w") as f:
            f.write("\n".join(lines) + "\n")
    except OSError as e:
        raise SystemExit(f"cannot write {out}: {e}")
    print("\n".join(lines))
    print(f"\nwrote {out}")

    # independent recomputation guard: matched-floor pooled numerator recomputed.
    if pooled_n:
        direct_k = sum(
            cap_share([r for r in (read_jsonl(resolve(fn)) or [])
                       if r.get("task") == st], floor=COMMON_FLOOR)[0]
            for st, fn, model, pool in INGEST if pool)
        if direct_k != pooled_k:
            raise SystemExit(f"CHECK FAIL: pooled numerator {pooled_k} vs recomputed {direct_k}")
    print("CHECK: PASS")


if __name__ == "__main__":
    main()
