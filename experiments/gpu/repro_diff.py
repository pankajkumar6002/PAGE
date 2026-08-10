"""Diff a fresh run against the released log, row for row.

Compares only the (task, id, budget) rows present in both files, so a reduced
rerun can be checked against a full log.

Fields are split by how much a mismatch matters:

  hard   correct_plain, correct_gated, gate_open, n_kept_plain, n_kept_gated
         Discrete outcomes. Under greedy decoding these must match exactly.
         Any mismatch means the environment changed the model's behaviour.

  soft   drop
         A float computed from attention. Tiny differences are expected from
         non-deterministic reductions on GPU; what matters is whether any
         difference is large enough to move an input across tau.
"""
import json
import sys
from collections import defaultdict

TAU = 0.07
HARD = ["correct_plain", "correct_gated", "gate_open", "n_kept_plain", "n_kept_gated"]


def load(path):
    out = {}
    with open(path) as f:
        for line in f:
            if not line.strip() or "\x00" in line:
                continue
            r = json.loads(line)
            out[(r["task"], r["id"], r["budget"])] = r
    return out


def main():
    new, ref = load(sys.argv[1]), load(sys.argv[2])
    shared = sorted(set(new) & set(ref))
    if not shared:
        print("FAIL: no overlapping (task, id, budget) rows")
        return 1
    print(f"comparing {len(shared)} rows "
          f"(fresh {len(new)}, reference {len(ref)})\n")

    mism = defaultdict(int)
    for k in shared:
        for f in HARD:
            if f in new[k] and f in ref[k] and new[k][f] != ref[k][f]:
                mism[f] += 1

    drops = [(abs(new[k]["drop"] - ref[k]["drop"]), k) for k in shared
             if "drop" in new[k] and "drop" in ref[k]]
    maxd, argmax = max(drops) if drops else (0.0, None)
    # a drop difference only matters if it moves an input across the threshold
    flips = sum(1 for k in shared
                if (new[k]["drop"] >= TAU) != (ref[k]["drop"] >= TAU))

    print("hard fields (must match exactly under greedy decoding):")
    for f in HARD:
        n = mism.get(f, 0)
        print(f"  {f:16s} {len(shared) - n:5d}/{len(shared)} match"
              + ("" if n == 0 else f"   <-- {n} MISMATCH"))
    print(f"\nsoft field  drop: max |delta| = {maxd:.3e}"
          + (f"  at {argmax}" if maxd > 1e-9 else ""))
    print(f"            gate decisions flipped across tau={TAU}: {flips}")

    ok = not mism and flips == 0
    print("\n" + ("PASS: environment reproduces the released log; seed "
                  "differences measured from here are attributable to the "
                  "input draw."
                  if ok else
                  "FAIL: environment does not reproduce the released log. "
                  "Seed variance would be confounded with this difference. "
                  "Do not run P3-1 until resolved."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())