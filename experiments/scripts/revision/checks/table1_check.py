"""Reproduces Table 1 and the pooled NIAH-MK3 count from the released logs. Usage: python3 -I table1_check.py"""
import collections
import json
import math
import os
from pathlib import Path

RES = os.environ.get("PAGE_RESULTS", str(Path(__file__).resolve().parents[3] / "results"))
T1 = {"vt": [.06, .19, .00, .05, .00, .03, .00, .32], "fwe": [.06, .04, .01, .32, .00, .20, .02, .22],
      "qa_1": [.08, .14, .03, .08, .06, .10, .04, .00], "qa_2": [.08, .05, .04, .02, .02, .03, .01, .00],
      "niah_multivalue": [.07, .10, .02, .13, .08, .07, .06, .21], "niah_multikey_3": [.01, .00, .00, .02, .00, .00, .00, .00]}
COLS = ["Qwen2.5-1.5B 4K", "Qwen2.5-1.5B 16K", "Qwen2.5-3B 4K", "Qwen2.5-3B 16K", "Qwen2.5-14B 4K", "Qwen2.5-14B 16K", "Mistral-7B 4K", "Mistral-7B 16K"]
SRC = [("ruler_vt_fwe_4k_snapkv.jsonl", 0, "sweep", None), ("ruler_mk3_4k_snapkv.jsonl", 0, "sweep", None),
       ("ruler_qa_4k_qwen.jsonl", 0, "sweep", None), ("ruler_extra_4k_qwen.jsonl", 0, "sweep", None),
       ("ruler_vt_fwe_16k_qwen.jsonl", 1, "sweep", None), ("ruler_mv_fwe_16k_qwen.jsonl", 1, "sweep", ["niah_multivalue"]),
       ("ruler_qa_16k_qwen.jsonl", 1, "sweep", None), ("ruler_niah_16k_qwen.jsonl", 1, "sweep", None),
       ("ruler_4k_qwen3b.jsonl", 2, "sweep", None), ("ruler_4k_qwen3b_mv_qa2.jsonl", 2, "sweep", None),
       ("ruler_16k_qwen3b.jsonl", 3, "sweep", None), ("ruler_16k_qwen3b_mvfwe.jsonl", 3, "sweep", None), ("ruler_16k_qwen3b_qa2.jsonl", 3, "sweep", None),
       ("gated_16k_qwen3b.jsonl", 3, "gated", ["niah_multikey_3"]),
       ("partition_qwen14b_4k.jsonl", 4, "partition", None), ("partition_qwen14b_16k.jsonl", 5, "partition", None),
       ("gated_4k_mistral7b.jsonl", 6, "gated", ["vt", "fwe", "qa_1", "niah_multikey_3"]), ("ruler_4k_mistral7b_qa2_mv.jsonl", 6, "sweep", None),
       ("gated_16k_mistral7b.jsonl", 7, "gated", ["vt", "fwe", "qa_1", "niah_multikey_3"]), ("ruler_16k_mistral7b_qa2_mv.jsonl", 7, "sweep", None)]


def ok(pred, golds, task):
    p = pred.lower()
    if task in ("vt", "fwe", "cwe", "niah_multivalue"):
        return all(g.strip().lower() in p for g in golds)
    return any(g.strip().lower() in p for g in golds)


def correct(r, kind, task):
    if kind == "sweep":
        return ok(r["pred"], r["gold"], task)
    if kind == "partition":
        return bool(r["plain_correct"])
    return bool(r["correct_plain"])


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


seen, bad_cells, mk_k, mk_n = set(), [], 0, 0
print("| cell | task | log | N | rho | Table 1 |\n|---|---|---|---|---|---|")
for fn, col, kind, only in SRC:
    by = collections.defaultdict(dict)
    for line in open(os.path.join(RES, fn)):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        by[(r["task"], r["id"])][float(r["budget"])] = r
    for t in sorted({t for t, _ in by}):
        if t not in T1 or (only and t not in only) or (col, t) in seen:
            continue
        seen.add((col, t))
        keys = [k for k in by if k[0] == t and 1.0 in by[k]]
        n = len(keys)
        k = sum(1 for kk in keys if not correct(by[kk][1.0], kind, t) and any(correct(by[kk][b], kind, t) for b in by[kk] if b < 1.0))
        want = T1[t][col]
        good = abs(round(k / n, 2) - want) < 0.006
        if not good:
            bad_cells.append((COLS[col], t))
        if t == "niah_multikey_3":
            mk_k, mk_n = mk_k + k, mk_n + n
        print(f"| {COLS[col]} | {t} | {fn} | {n} | {k / n:.2f} | {want:.2f}{'' if good else ' DIFFERS'} |")
lo, hi = wilson(mk_k, mk_n)
print(f"\nCells reproduced: {len(seen) - len(bad_cells)} of {len(seen)} (Table 1 has 48).")
print(f"NIAH-MK3 pooled: {mk_k}/{mk_n}, rho = {mk_k / mk_n:.3f}, Wilson [{lo:.3f}, {hi:.3f}] (paper: 2/526, 0.004, [0.001, 0.014]).")
print("PASS" if not bad_cells and len(seen) == 48 and (mk_k, mk_n) == (2, 526) else f"FAIL {bad_cells}")
