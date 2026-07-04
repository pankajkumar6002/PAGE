import json, sys, re
from collections import defaultdict

path = sys.argv[1]
tau = 0.07

rows = [json.loads(l) for l in open(path) if l.strip()]

def exact_int(pred, golds):
    # first integer in pred vs gold integer(s)
    m = re.search(r"-?\d+", pred)
    if not m:
        return False
    p = m.group(0)
    return any(p == g.strip() for g in golds if g.strip())

# group by id
by_id = defaultdict(dict)   # id -> {budget: rec}
for r in rows:
    by_id[r["id"]][r["budget"]] = r

ids = sorted(by_id)
budgets = sorted({r["budget"] for r in rows}, reverse=True)
N = len(ids)
print(f"file={path}")
print(f"N inputs = {N}, budgets = {budgets}")

# accuracy vs budget, substring (repo) and exact-int
for metric_name, use_exact in [("substring(repo)", False), ("exact-int", True)]:
    print(f"\n== metric: {metric_name} ==")
    print("budget | plain acc | gated acc")
    for b in budgets:
        cp = cg = n = 0
        for i in ids:
            r = by_id[i].get(b)
            if r is None:
                continue
            n += 1
            if use_exact:
                cp += exact_int(r["pred_plain"], r["answers"])
                cg += exact_int(r["pred_gated"], r["answers"])
            else:
                cp += r["correct_plain"]
                cg += r["correct_gated"]
        print(f"{b:>6} | {cp/n:.3f}   | {cg/n:.3f}   (n={n})")

    # rho: fraction of inputs where full-KV (b=1.0) wrong AND some b<1 correct (plain)
    def correct(r):
        return exact_int(r["pred_plain"], r["answers"]) if use_exact else r["correct_plain"]
    rho_num = 0
    afull = 0
    for i in ids:
        r1 = by_id[i].get(1.0)
        if r1 is None:
            continue
        full_ok = correct(r1)
        afull += full_ok
        if not full_ok:
            if any(correct(by_id[i][b]) for b in budgets if b < 1.0 and b in by_id[i]):
                rho_num += 1
    print(f"A_full = {afull/N:.3f}   rho = {rho_num/N:.3f} ({rho_num}/{N})")

# gate / drop stats (per-input, one drop per id)
drops = [by_id[i][1.0]["drop"] for i in ids if 1.0 in by_id[i]]
import statistics as st
below = sum(1 for d in drops if d < tau)
print(f"\n== gate / drop ==")
print(f"mean D = {st.mean(drops):.4f}  std {st.pstdev(drops):.4f}  min {min(drops):.4f}  max {max(drops):.4f}")
print(f"inputs with D < tau({tau}): {below}/{len(drops)} = {below/len(drops):.3f}")
print(f"gate-open fraction (D>=tau): {1-below/len(drops):.3f}")

# T range
Ts = [by_id[i][1.0]["T"] for i in ids if 1.0 in by_id[i]]
print(f"T range: {min(Ts)}-{max(Ts)}")
