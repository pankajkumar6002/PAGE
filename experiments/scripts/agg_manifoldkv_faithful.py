"""Aggregate accuracy vs budget per (scorer, alloc) for the faithful ManifoldKV run."""
import json
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "experiments/results/manifoldkv_faithful_mk3.jsonl"
rows = [json.loads(l) for l in open(path)]

# full-KV baseline (logged once per scorer under alloc="full")
full = defaultdict(list)
cells = defaultdict(list)  # (scorer, alloc, budget) -> [correct]
eff = defaultdict(list)
for r in rows:
    if r["budget"] >= 1.0:
        full[r["scorer"]].append(r["correct"])
    else:
        key = (r["scorer"], r["alloc"], r["budget"])
        cells[key].append(r["correct"])
        eff[key].append(r["eff_budget"])

scorers = ["snapkv", "keydiff", "manifoldkv", "manifoldkv_pre"]
allocs = ["uniform", "adakv"]
budgets = sorted({r["budget"] for r in rows if r["budget"] < 1.0}, reverse=True)
N = max(len(v) for v in full.values())
print(f"N={N} examples  task={rows[0]['task']}\n")

# full baseline
print("Full-KV (b=1.0) accuracy per scorer:")
for s in scorers:
    if full.get(s):
        print(f"  {s:16s} {sum(full[s])/len(full[s]):.3f}")
print()

hdr = "scorer/alloc      " + "".join(f"b={b:<7}" for b in budgets)
for s in scorers:
    print(f"--- {s} ---")
    for a in allocs:
        line = f"  {a:14s}"
        for b in budgets:
            v = cells.get((s, a, b), [])
            acc = sum(v)/len(v) if v else float("nan")
            line += f"{acc:<9.3f}"
        print(line)
    print()

print("Mean effective budget (adakv, manifoldkv) sanity:")
for b in budgets:
    v = eff.get(("manifoldkv", "adakv", b), [])
    if v:
        print(f"  nominal {b}: eff={sum(v)/len(v):.4f}")
