"""Actually fit the scaling-figure slope.

The paper's scaling figure carries three different numbers for one quantity:
a caption stating "a band of slope approx 0.4", body text stating "the ratio
clusters in [0.25, 0.55]", and no OLS fit anywhere to justify either. There is
also an unexplained outlier: Qwen2.5-3B 4K FWE has ratio 0.04, an order of
magnitude below the claimed band, with real headroom (0.24).

This script fits the through-origin OLS the figure claims to show. Data are
transcribed from the paper's scaling-formula table; the row with no headroom
(Qwen2.5-3B 4K VT, where 1 - A_full = 0.00 and the ratio is undefined) is
dropped, as that table does.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paths import out_path

# (label, 1 - A_full, rho, is_capacity_bound). Transcribed from the paper's
# scaling-formula table.
ROWS = [
    ("Qwen2.5-1.5B 4K  VT",       0.18, 0.06, False),
    ("Qwen2.5-1.5B 16K VT",       0.75, 0.19, False),
    ("Qwen2.5-3B   16K VT",       0.09, 0.05, False),
    ("Qwen2.5-3B   4K  FWE",      0.24, 0.01, False),
    ("Qwen2.5-3B   16K FWE",      0.62, 0.32, False),
    ("Qwen2.5-1.5B 4K  NIAH-MK3", 0.35, 0.01, True),
    ("Qwen2.5-1.5B 16K NIAH-MK3", 0.74, 0.00, True),
]
# Dropped: Qwen2.5-3B 4K VT, headroom 0.00, ratio undefined (the table prints "--").

CLAIMED_BAND = (0.25, 0.55)
CLAIMED_SLOPE = 0.4
TOL = 1e-4

EXPECTED_DIL = 0.3432
EXPECTED_ALL = 0.2111


def ols_through_origin(pts):
    """slope minimising sum (y - m x)^2 is sum(xy)/sum(x^2)."""
    sxy = sum(x * y for x, y, in pts)
    sxx = sum(x * x for x, _ in pts)
    return sxy / sxx if sxx else float("nan")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    dil = [(h, r) for _, h, r, cb in ROWS if not cb]
    allr = [(h, r) for _, h, r, _ in ROWS]

    m_dil = ols_through_origin(dil)
    m_all = ols_through_origin(allr)

    lines = [
        "# The scaling-figure slope, actually fitted",
        "",
        "The scaling figure reports a slope, its caption reports a different",
        "one, and the surrounding text reports a band. Nothing in the paper is",
        "fitted: the string \"OLS\" does not appear. Below is the through-origin",
        "fit the figure claims to display.",
        "",
        "| population | n | through-origin OLS slope |",
        "|---|---:|---:|",
        f"| dilution-prone only (what the figure labels) | {len(dil)} | **{m_dil:.4f}** |",
        f"| including NIAH-MK3 | {len(allr)} | {m_all:.4f} |",
        "",
        "| number in the paper | value | where |",
        "|---|---:|---|",
        f"| figure caption \"band of slope\" | {CLAIMED_SLOPE:.2f} | scaling figure caption |",
        f"| text \"ratio clusters in\" | {CLAIMED_BAND[0]:.2f}-{CLAIMED_BAND[1]:.2f} | body text |",
        f"| **fitted, dilution-prone** | **{m_dil:.2f}** | this script |",
        "",
        "## Per-row ratios, and the outlier the band does not contain",
        "",
        "| cell | 1 - A_full | rho | ratio | in [0.25, 0.55]? |",
        "|---|---:|---:|---:|---|",
    ]
    outliers = []
    for label, h, r, cb in ROWS:
        ratio = r / h if h else float("nan")
        if not cb:
            inside = CLAIMED_BAND[0] <= ratio <= CLAIMED_BAND[1]
            if not inside:
                outliers.append((label, ratio))
            mark = "yes" if inside else "**NO**"
        else:
            mark = "n/a (capacity-bound)"
        lines.append(f"| {label} | {h:.2f} | {r:.2f} | {ratio:.3f} | {mark} |")

    lo, hi = wilson(1, 100)
    lines += [
        "",
        "## What this settles",
        "",
        f"The fitted dilution-prone slope is **{m_dil:.2f}**, not the "
        f"{CLAIMED_SLOPE:.1f} the caption asserts.",
        f"Including the capacity-bound rows gives {m_all:.2f}, which is close to",
        "a value that would be reported if the fit had accidentally been run over",
        "the whole table rather than the population the figure labels.",
        "",
        "**The band has one clear counterexample and one boundary case.** The",
        "boundary case is Qwen2.5-3B 16K VT: the printed 0.05/0.09 gives 0.556,",
        "just above the stated 0.55, but both inputs are 2-dp so the true ratio",
        "spans roughly [0.474, 0.647]. It straddles the edge and cannot be",
        "called in or out from the published precision. The clear counterexample",
        "is Qwen2.5-3B 4K FWE, which",
        f"sits at ratio {0.01/0.24:.3f} with real headroom (0.24). At N = 100 an",
        f"observed rho = 0.01 is a single recovery, Wilson [{lo:.3f}, {hi:.3f}],",
        "so the ratio is dominated by counting noise rather than by the",
        "relationship being modelled.",
        "",
        "**Two defensible fixes, pick one and say which:**",
        "",
        f"1. Replace \"slope ~{CLAIMED_SLOPE:.1f}\" with the fitted {m_dil:.2f}, and",
        "   exclude the FWE-4K cell from the band with the counting-noise reason",
        "   stated. The band then honestly describes 4 of 5 dilution rows.",
        f"2. Keep every row and widen the band to [{0.01/0.24:.2f}, 0.55]. This is",
        "   more honest but the band no longer supports \"approximately constant\",",
        "   so the surrounding prose must weaken accordingly.",
        "",
        "Either way, the paper must stop printing three different numbers for one",
        "slope; it is a two-minute recomputation to check.",
    ]

    ok = True
    checks = [
        (f"dilution-only slope == {EXPECTED_DIL}", abs(m_dil - EXPECTED_DIL) <= TOL,
         f"{m_dil:.4f}"),
        (f"with-MK3 slope == {EXPECTED_ALL}", abs(m_all - EXPECTED_ALL) <= TOL,
         f"{m_all:.4f}"),
        ("fitted slope differs from the caption's 0.4",
         abs(m_dil - CLAIMED_SLOPE) > 0.02, f"|{m_dil:.4f} - 0.4| = {abs(m_dil-CLAIMED_SLOPE):.4f}"),
        # FWE-4K (0.042) is unambiguously outside. 16K VT computes to 0.556 but
        # its 2-dp inputs admit [0.474, 0.647], so it is a boundary case, not a
        # counterexample. Assert only the unambiguous one.
        ("FWE-4K falls outside the claimed band",
         any(abs(r - 0.042) < 5e-3 for _, r in outliers),
         f"{len(outliers)} outside: {[round(r,3) for _, r in outliers]}"),
    ]
    for _, passed, _ in checks:
        ok &= passed

    lines += ["", "## Checks", ""]
    for name, passed, got in checks:
        lines.append(f"- [{'PASS' if passed else 'FAIL'}] {name} (got {got})")
    for label, ratio in outliers:
        lines.append(f"  - outside band: {label} at {ratio:.3f}")

    path = out_path("scaling_slope_fit.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {path}")
    print("CHECK: PASS" if ok else "CHECK: FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()