"""The headline matrix restricted to aggressive budgets.

The paper's headline-matrix caption disowns its own moderate-budget regime
("a clean per-head SnapKV collapses less sharply at 2x ... read them at >=
4x"), yet the reported grand mean averages over every b < 1.0, which includes
b = 0.875, 0.75, 0.625 and 0.5. This recomputes the matrix restricted to
b <= 0.25, the regime the caption actually endorses.

The restriction RAISES the grand mean, from +0.2286 to +0.2584: restricting to
the aggressive regime does not deflate the headline as might be suspected,
because the collapse the gate prevents is deepest exactly where the caption
says to read it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paths import MODELS_HF as MODELS, TAU, MK3, released_cell, out_path
from gatelib import eviction_rows, delta

POLICIES = ["snapkv", "h2o", "streamingllm", "pyramidkv"]

UNRESTRICTED = lambda b: b < 1.0
RESTRICTED = lambda b: b <= 0.25

# The released cells do NOT share a budget grid: SnapKV cells carry 8 budgets
# (0.0625..0.875) while h2o/streamingllm/pyramidkv carry only {0.125,0.25,0.5}.
# Restricting to b<=0.25 therefore drops 5 of 8 budgets on SnapKV cells but
# only 1 of 3 elsewhere, so a raw before/after comparison confounds the
# restriction with the grid. MATCHED_* compares on the budgets every cell has.
MATCHED = {0.125, 0.25, 0.5}
MATCHED_UNRESTRICTED = lambda b: b in MATCHED
MATCHED_RESTRICTED = lambda b: b in MATCHED and b <= 0.25
EXPECTED_MATCHED_UNRES = 0.2402
EXPECTED_MATCHED_RES = 0.2569

EXPECTED_UNRESTRICTED = 0.2286
EXPECTED_RESTRICTED = 0.2584
TOL = 1e-3


def cells(budget_filter):
    """(policy, slug) -> delta, over the cells whose logs exist."""
    out = {}
    for policy in POLICIES:
        for _, slug, _ in MODELS:
            path = released_cell(slug, policy)
            if not os.path.exists(path):
                continue
            rows = eviction_rows(path, policy=policy, slug=slug, tau=TAU,
                                 budget_filter=budget_filter)
            if not rows:
                continue
            out[(policy, slug)] = {
                "all": delta(rows),
                "mk3": delta(rows, lambda t: t == MK3),
                "no_mk3": delta(rows, lambda t: t != MK3),
                "n": len(rows),
            }
    return out


def main():
    unres = cells(UNRESTRICTED)
    res = cells(RESTRICTED)
    shared = sorted(set(unres) & set(res))
    if not shared:
        raise SystemExit("no cells found; set PAGE_RESULTS")

    gm_unres = sum(unres[k]["all"] for k in shared) / len(shared)
    gm_res = sum(res[k]["all"] for k in shared) / len(shared)
    mu = cells(MATCHED_UNRESTRICTED); mr = cells(MATCHED_RESTRICTED)
    msh = sorted(set(mu) & set(mr))
    gm_mu = sum(mu[k]["all"] for k in msh) / len(msh)
    gm_mr = sum(mr[k]["all"] for k in msh) / len(msh)
    gm_res_nomk3 = sum(res[k]["no_mk3"] for k in shared) / len(shared)

    lines = [
        "# Headline matrix restricted to b <= 0.25",
        "",
        f"tau = {TAU}. Delta = gated minus plain accuracy, over the same inputs.",
        "The `b < 1.0` column is the convention tab:matrix currently reports;",
        "`b <= 0.25` is the >= 4x regime its own caption directs the reader to.",
        "",
        "| cell | Delta (b<1.0) | Delta (b<=0.25) | change | MK3 (b<=0.25) | no-MK3 (b<=0.25) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    regressions = []
    for policy, slug in shared:
        u = unres[(policy, slug)]["all"]
        r = res[(policy, slug)]["all"]
        if r < u - TOL:
            regressions.append((policy, slug, u, r))
        lines.append(
            f"| {policy} {slug} | {u:+.3f} | {r:+.3f} | {r - u:+.3f} | "
            f"{res[(policy, slug)]['mk3']:+.3f} | {res[(policy, slug)]['no_mk3']:+.3f} |")
    lines += [
        f"| **grand mean** | **{gm_unres:+.4f}** | **{gm_res:+.4f}** | "
        f"**{gm_res - gm_unres:+.4f}** | | **{gm_res_nomk3:+.4f}** |",
        "",
        f"Cells: {len(shared)}. Restricting to the aggressive regime moves the "
        f"grand mean {gm_unres:+.4f} -> {gm_res:+.4f} "
        f"({gm_res - gm_unres:+.4f}).",
        "",
        "## Matched-grid control (the honest comparison)",
        "",
        "The cells do not share a budget grid: SnapKV carries 8 budgets, the",
        "other three policies only {0.125, 0.25, 0.5}. The raw shift above",
        "therefore mixes the restriction with the grid. Restricted to the three",
        "budgets every cell has:",
        "",
        "| grid | grand mean |",
        "|---|---:|",
        f"| {{0.125, 0.25, 0.5}} | {gm_mu:+.4f} |",
        f"| {{0.125, 0.25}} (aggressive) | {gm_mr:+.4f} |",
        f"| **restriction effect** | **{gm_mr - gm_mu:+.4f}** |",
        "",
        "The direction survives the control: restricting to the aggressive",
        "budgets raises the mean on a like-for-like grid too. Quote the",
        f"matched-grid effect ({gm_mr - gm_mu:+.4f}), not the raw one.",
        "",
        "## What this settles",
        "",
        "Restricting the headline to the aggressive (>= 4x) budget regime does",
        "not deflate it, as averaging over the disowned moderate-budget regime",
        "might be suspected to do. The restricted mean is HIGHER, so the",
        "reported +22.9pp is if anything conservative with respect to the",
        "budgets the paper's own caption endorses.",
        "",
        "This does not by itself answer the baseline objection: both columns",
        "use the shared single keep-mask. The per-head Ada-KV analysis",
        "(`adakv_matrix.md`) replaces the plain arm with per-head allocation",
        "and is the experiment that settles that question.",
        "",
        "## Which 'clean per-head SnapKV' figure is meant",
        "",
        "Two different per-head allocations of the same SnapKV scores appear",
        "elsewhere in the paper: **0.32 is the Ada-KV per-head row**, and",
        "**0.20 is the uniform per-head row**. Any citation of \"clean per-head",
        "SnapKV\" should name which one it means.",
    ]

    ok = True
    checks = []
    d = abs(gm_unres - EXPECTED_UNRESTRICTED)
    checks.append((f"grand mean b<1.0 == {EXPECTED_UNRESTRICTED}", d <= TOL, f"{gm_unres:.4f}"))
    ok &= d <= TOL
    d = abs(gm_res - EXPECTED_RESTRICTED)
    checks.append((f"grand mean b<=0.25 == {EXPECTED_RESTRICTED}", d <= TOL, f"{gm_res:.4f}"))
    ok &= d <= TOL
    for nm, val, exp in [("matched-grid b in {0.125,0.25,0.5}", gm_mu, EXPECTED_MATCHED_UNRES),
                         ("matched-grid b in {0.125,0.25}", gm_mr, EXPECTED_MATCHED_RES)]:
        d2 = abs(val - exp)
        checks.append((f"{nm} == {exp}", d2 <= TOL, f"{val:.4f}"))
        ok &= d2 <= TOL
    d2 = gm_mr - gm_mu
    checks.append(("restriction raises the mean on a matched grid", d2 > 0, f"{d2:+.4f}"))
    ok &= d2 > 0
    no_reg = not regressions
    checks.append(("every restricted cell >= its unrestricted value", no_reg,
                   "ok" if no_reg else f"{len(regressions)} regressed"))
    ok &= no_reg

    lines += ["", "## Checks", ""]
    for name, passed, got in checks:
        lines.append(f"- [{'PASS' if passed else 'FAIL'}] {name} (got {got})")
    if regressions:
        lines.append("")
        for policy, slug, u, r in regressions:
            lines.append(f"  - regressed: {policy} {slug} {u:+.4f} -> {r:+.4f}")

    path = out_path("matrix_budget_restricted.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {path}")
    print("CHECK: PASS" if ok else "CHECK: FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()