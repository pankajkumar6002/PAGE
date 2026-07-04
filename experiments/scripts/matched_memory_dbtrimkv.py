"""Part 2 aggregation: plain DBTrimKV accuracy vs kept-KV (old 128/256/512 +
new 1024/1152/1280), and the matched-memory head-to-head at ~30% kept."""
import json, collections, statistics
import numpy as np

R = "/home/smlab/projects/eff-nn/experiments/results/"
NHEAD = None  # infer from n_kept/ memory


def kept_frac(r):
    # per-head kept = n_kept_plain / (L*kv_heads); frac vs T. For Qwen3-4B L*heads=288.
    return (r["n_kept_plain"] / 288) / r["T"]


def agg(recs):
    by = collections.defaultdict(list)
    for r in recs:
        by[r["budget"]].append(r)
    out = []
    for m in sorted(by):
        rs = by[m]
        kf = statistics.mean(kept_frac(r) for r in rs)
        ac = statistics.mean(bool(r["correct_plain"]) for r in rs)
        # per task
        pt = {}
        bt = collections.defaultdict(list)
        for r in rs:
            bt[r["task"]].append(bool(r["correct_plain"]))
        for t, v in bt.items():
            pt[t] = statistics.mean(v)
        out.append((m, len(rs), kf, ac, pt))
    return out


old = [json.loads(l) for l in open(R + "gated_dbtrimkv_qwen3_4k.jsonl") if l.strip()]
new = [json.loads(l) for l in open(R + "plain_dbtrimkv_qwen3_4k_30pct.jsonl") if l.strip()]

print("=== plain DBTrimKV curve (old M=128/256/512) ===")
for m, n, kf, ac, pt in agg(old):
    print(f"  M={m:<5} N={n} kept={kf:.3f} acc={ac:.3f}  per-task={ {k:round(v,2) for k,v in pt.items()} }")
print("=== plain DBTrimKV curve (NEW M=1024/1152/1280) ===")
newagg = agg(new)
for m, n, kf, ac, pt in newagg:
    print(f"  M={m:<5} N={n} kept={kf:.3f} acc={ac:.3f}  per-task={ {k:round(v,2) for k,v in pt.items()} }")

# full curve for interpolation at 0.30
allpts = [(kf, ac) for m, n, kf, ac, pt in agg(old + new)]
allpts = sorted(allpts)
xs = np.array([p[0] for p in allpts]); ys = np.array([p[1] for p in allpts])
db30 = float(np.interp(0.30, xs, ys))
print(f"\n  full DBTrimKV curve: {[(round(k,3),round(a,3)) for k,a in allpts]}")
print(f"  plain DBTrimKV acc @ 30% kept (interp) = {db30:.3f}")
# the M closest to 30%
closest = min(newagg, key=lambda x: abs(x[2] - 0.30))
print(f"  plain DBTrimKV closest single budget to 30%: M={closest[0]} kept={closest[2]:.3f} acc={closest[3]:.3f} (N={closest[1]})")
