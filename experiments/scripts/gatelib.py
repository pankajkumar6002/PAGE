"""Shared loading logic for the revision-round reanalyses.

The one non-obvious piece is the post-hoc threshold identity from App.
`app:repro`: decoding is greedy, so for any threshold tau the gated outcome of
an input is determined by its recorded drop plus two stored outcomes,

    correct_gated(tau) = correct_plain(b)   if drop >= tau   (gate opens)
                       = correct_plain(1.0) otherwise        (gate closes)

Every script here re-evaluates at tau=0.07 through this identity rather than
reading `correct_gated` off the log, so the one cell executed at tau=0.04
(Qwen2.5-3B 4K SnapKV) is comparable with the other fifteen. The
reconstruction is asserted exact on every cell that *was* executed at 0.07.
"""
import json

from paths import TAU, RECONSTRUCTED


def read_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def full_cache_outcomes(rows):
    """(task, id) -> full-cache correctness, from the b=1.0 rows."""
    return {(r["task"], r["id"]): bool(r["correct_plain"])
            for r in rows if r["budget"] == 1.0}


def eviction_rows(path, policy=None, slug=None, tau=TAU, budget_filter=None,
                  stored=False):
    """Rows at eviction budgets with `gated` re-evaluated at `tau`.

    budget_filter defaults to b < 1.0, the convention pinned by
    method_agnostic_matrix.py and the tab:matrix caption.

    stored=True reads `correct_gated` off the log instead of re-evaluating.
    That mixes thresholds across cells, since the Qwen2.5-3B 4K cell was
    executed at tau=0.04, so it is only for reproducing earlier analyses.
    """
    if budget_filter is None:
        budget_filter = lambda b: b < 1.0
    rows = read_rows(path)
    full = full_cache_outcomes(rows)
    verify = not stored and (policy, slug) not in RECONSTRUCTED
    out, checked, mismatch = [], 0, 0
    for r in rows:
        if not budget_filter(r["budget"]) or r["budget"] >= 1.0:
            continue
        key = (r["task"], r["id"])
        if key not in full:
            continue
        opened = r["drop"] >= tau
        if stored:
            gated = bool(r["correct_gated"])
        else:
            gated = bool(r["correct_plain"]) if opened else full[key]
        if verify:
            checked += 1
            mismatch += int(gated != bool(r["correct_gated"]))
        out.append({
            "task": r["task"], "id": r["id"], "budget": r["budget"],
            "drop": r["drop"], "open": opened,
            "plain": bool(r["correct_plain"]), "gated": gated, "full": full[key],
        })
    if verify and mismatch:
        raise SystemExit(
            f"FAIL: tau={tau} reconstruction disagrees with stored correct_gated on "
            f"{mismatch}/{checked} rows of {path}. The post-hoc identity in "
            f"App. app:repro does not hold for this cell; every downstream number "
            f"that re-evaluates the threshold is invalid."
        )
    return out


def rate(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def delta(rows, pred=lambda t: True):
    sel = [r for r in rows if pred(r["task"])]
    if not sel:
        return float("nan")
    return rate([r["gated"] for r in sel]) - rate([r["plain"] for r in sel])
