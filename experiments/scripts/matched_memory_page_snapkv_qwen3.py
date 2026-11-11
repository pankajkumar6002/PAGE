"""PAGE-SnapKV on Qwen3-4B: achieved kept-KV and accuracy vs (tau, budget).

Shows (a) at the paper's tau=0.07 PAGE cannot compress (floor ~0.99), and
(b) the (tau,b) needed to force ~30% ACHIEVED kept, and the accuracy there.
Restricted to the 120-input matched suite (first 30/task) that DBTrimKV ran.
"""
import json
from collections import defaultdict
import numpy as np

RES = "experiments/results/gated_4k_qwen3_4b_merged.jsonl"
# per-task index < 30  ->  global id ranges matching DBTrimKV's first-30/task suite
OFFSET = {"niah_multikey_3": 0, "vt": 100, "fwe": 200, "qa_1": 300}


def load(matched=True):
    recs = [json.loads(l) for l in open(RES) if l.strip()]
    by_id = defaultdict(dict)
    for r in recs:
        if matched and (r["id"] - OFFSET[r["task"]]) >= 30:
            continue
        by_id[(r["id"], r["task"])][r["budget"]] = r
    full = {}
    for k, bm in by_id.items():
        full[k] = bool(bm[1.0]["correct_plain"])
    return by_id, full


def page_point(by_id, full, tau, b):
    kept, acc = [], []
    for k, bm in by_id.items():
        r = bm[b]
        opened = r["drop"] >= tau
        if opened:
            kept.append(r["n_kept_plain"] / r["T"]); acc.append(bool(r["correct_plain"]))
        else:
            kept.append(1.0); acc.append(full[k])
    return float(np.mean(kept)), float(np.mean(acc)), float(np.mean([bm[b]["drop"] >= tau for bm in by_id.values()]))


def plain_curve(by_id):
    budgets = sorted({b for bm in by_id.values() for b in bm})
    pts = []
    for b in budgets:
        kf = np.mean([bm[b]["n_kept_plain"] / bm[b]["T"] for bm in by_id.values()])
        ac = np.mean([bool(bm[b]["correct_plain"]) for bm in by_id.values()])
        pts.append((float(kf), float(ac)))
    return sorted(pts)


def main():
    for matched, label in [(True, "MATCHED 120-input suite (first 30/task, == DBTrimKV suite)"),
                           (False, "FULL 400-input suite (100/task)")]:
        by_id, full = load(matched)
        print("=" * 90)
        print(f"PAGE-SnapKV Qwen3-4B  --  {label}  (N={len(by_id)}, A_full={np.mean(list(full.values())):.3f})")
        budgets = sorted({b for bm in by_id.values() for b in bm if b < 1.0})
        print("\n  At paper operating tau=0.07:")
        for b in budgets:
            k, a, op = page_point(by_id, full, 0.07, b)
            print(f"    b={b:<7} achieved_kept={k:.3f} acc={a:.3f} gate_open={op:.3f}")
        print("\n  Sweep (tau,b) to reach ~30% achieved kept:")
        best = None
        for tau in np.arange(0.010, 0.075, 0.0025):
            for b in budgets:
                k, a, op = page_point(by_id, full, float(tau), b)
                if 0.27 <= k <= 0.33:
                    if best is None or abs(k - 0.30) < abs(best[0] - 0.30):
                        best = (k, a, float(tau), b, op)
        if best:
            k, a, tau, b, op = best
            print(f"    closest to 0.30: tau={tau:.4f} b={b} -> achieved_kept={k:.3f} "
                  f"acc={a:.3f} (gate_open={op:.3f})")
        # a few representative points near 30%
        print("    representative points near 30% kept:")
        for tau, b in [(0.020, 0.0625), (0.020, 0.125), (0.025, 0.0625), (0.030, 0.0625), (0.030, 0.125)]:
            k, a, op = page_point(by_id, full, tau, b)
            print(f"      tau={tau:.3f} b={b:<7} kept={k:.3f} acc={a:.3f} open={op:.3f}")
        # plain SnapKV reference at ~30% kept
        pc = plain_curve(by_id)
        xs = np.array([p[0] for p in pc]); ys = np.array([p[1] for p in pc])
        plain30 = float(np.interp(0.30, xs, ys))
        print(f"\n  plain SnapKV curve: {[(round(k,3),round(a,3)) for k,a in pc]}")
        print(f"  plain SnapKV acc @ 30% kept (interp) = {plain30:.3f}")
        print()


if __name__ == "__main__":
    main()
