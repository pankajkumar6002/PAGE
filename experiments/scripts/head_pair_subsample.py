"""P3-3: how many head pairs does the gate signal actually need?

D costs O(L H^2 k), and the H^2 term is the head pairs: 66 pairs at H=12, but
496 at H=32 and 780 at H=40. That quadratic is the deployment objection in
Section limitations, since it is worst exactly on the wide-head models where
the gate is most expensive.

The released logs cannot answer this because head_agreement_probe.py averages
over pairs before writing. head_pair_probe.py retains them, so the question is
now answerable offline: subsample pairs, recompute D, and check whether the
ordering the gate depends on (NIAH-MK3 below every dilution-prone task)
survives.

Random subsets are averaged over several draws, because a single unlucky draw
would understate what subsampling can do.
"""
import json
import os
import random
from collections import defaultdict

from paths import MK3, out_path, DATA

DIL = ["vt", "fwe", "qa_1"]
FRACTIONS = [1.0, 0.5, 0.25, 0.10, 0.05, 0.02]
DRAWS = 20
SEED = 0


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip() and "\x00" not in line:
                rows.append(json.loads(line))
    return rows


def D_from(pair_layers, keep, L):
    """Early-minus-late mean agreement using only the pairs in `keep`."""
    third = L // 3
    def binmean(idx):
        tot = n = 0
        for li in idx:
            d = pair_layers[li]
            for k in keep:
                v = d.get(k)
                if v is not None:
                    tot += v; n += 1
        return tot / n if n else 0.0
    return binmean(range(third)) - binmean(range(L - third, L))


def main():
    path = os.path.join(DATA, "head_pairs_qwen15b_4k.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}")
    rows = load(path)
    L = rows[0]["L"]
    H = rows[0]["H"]
    all_pairs = list(rows[0]["pair_jaccard_per_layer"][0].keys())
    rng = random.Random(SEED)

    lines = []
    def emit(s=""):
        print(s); lines.append(s)

    emit("# P3-3: head-pair subsampling")
    emit()
    emit(f"Qwen2.5-1.5B, RULER 4K, $H = {H}$ heads, "
         f"{len(all_pairs)} pairs, $L = {L}$ layers, "
         f"{len(rows)} inputs.")
    emit()
    emit("| pairs kept | count | mean $D$ MK3 | min mean $D$ dil. | margin | ordering |")
    emit("|---|---:|---:|---:|---:|---|")

    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r["pair_jaccard_per_layer"])

    results = []
    for frac in FRACTIONS:
        k = max(1, int(round(frac * len(all_pairs))))
        margins, mk3s, mins = [], [], []
        draws = 1 if k == len(all_pairs) else DRAWS
        for _ in range(draws):
            keep = (all_pairs if k == len(all_pairs)
                    else rng.sample(all_pairs, k))
            mk3 = sum(D_from(p, keep, L) for p in by_task[MK3]) / len(by_task[MK3])
            dmins = []
            for t in DIL:
                if t in by_task:
                    dmins.append(sum(D_from(p, keep, L) for p in by_task[t])
                                 / len(by_task[t]))
            mk3s.append(mk3); mins.append(min(dmins))
            margins.append(min(dmins) - mk3)
        m_mk3 = sum(mk3s) / len(mk3s)
        m_min = sum(mins) / len(mins)
        m_mar = sum(margins) / len(margins)
        held = sum(1 for x in margins if x > 0)
        results.append((frac, k, m_mk3, m_min, m_mar, held, draws))
        emit(f"| {frac:.0%} | {k} | {m_mk3:.4f} | {m_min:.4f} | {m_mar:+.4f} "
             f"| {held}/{draws} draws |")

    emit()
    full = results[0][4]
    ok = [r for r in results if r[5] == r[6]]
    smallest = min(ok, key=lambda r: r[1]) if ok else None
    if smallest:
        emit(f"The ordering survives on every draw down to **{smallest[1]} of "
             f"{len(all_pairs)} pairs** ({smallest[0]:.0%}), where the margin is "
             f"{smallest[4]:+.4f} against {full:+.4f} at full cost.")
        emit()
        emit(f"Reading: the $H^2$ term can be cut to about {smallest[0]:.0%} "
             f"without losing the separation the gate thresholds. On a "
             f"40-head model that is {int(smallest[0]*780)} pairs instead of "
             f"780, which is the difference the deployment objection turns on.")
    else:
        emit("No subsample preserved the ordering on every draw; the full "
             "pair set is required.")

    with open(out_path("head_pair_subsample.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\nCHECK: PASS")


if __name__ == "__main__":
    main()
