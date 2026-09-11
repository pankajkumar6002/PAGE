# The scaling-figure slope, actually fitted

The scaling figure reports a slope, its caption reports a different
one, and the surrounding text reports a band. Nothing in the paper is
fitted: the string "OLS" does not appear. Below is the through-origin
fit the figure claims to display.

| population | n | through-origin OLS slope |
|---|---:|---:|
| dilution-prone only (what the figure labels) | 5 | **0.3432** |
| including NIAH-MK3 | 7 | 0.2111 |

| number in the paper | value | where |
|---|---:|---|
| figure caption "band of slope" | 0.40 | scaling figure caption |
| text "ratio clusters in" | 0.25-0.55 | body text |
| **fitted, dilution-prone** | **0.34** | this script |

## Per-row ratios, and the outlier the band does not contain

| cell | 1 - A_full | rho | ratio | in [0.25, 0.55]? |
|---|---:|---:|---:|---|
| Qwen2.5-1.5B 4K  VT | 0.18 | 0.06 | 0.333 | yes |
| Qwen2.5-1.5B 16K VT | 0.75 | 0.19 | 0.253 | yes |
| Qwen2.5-3B   16K VT | 0.09 | 0.05 | 0.556 | **NO** |
| Qwen2.5-3B   4K  FWE | 0.24 | 0.01 | 0.042 | **NO** |
| Qwen2.5-3B   16K FWE | 0.62 | 0.32 | 0.516 | yes |
| Qwen2.5-1.5B 4K  NIAH-MK3 | 0.35 | 0.01 | 0.029 | n/a (capacity-bound) |
| Qwen2.5-1.5B 16K NIAH-MK3 | 0.74 | 0.00 | 0.000 | n/a (capacity-bound) |

## What this settles

The fitted dilution-prone slope is **0.34**, not the 0.4 the caption asserts.
Including the capacity-bound rows gives 0.21, which is close to
a value that would be reported if the fit had accidentally been run over
the whole table rather than the population the figure labels.

**The band has one clear counterexample and one boundary case.** The
boundary case is Qwen2.5-3B 16K VT: the printed 0.05/0.09 gives 0.556,
just above the stated 0.55, but both inputs are 2-dp so the true ratio
spans roughly [0.474, 0.647]. It straddles the edge and cannot be
called in or out from the published precision. The clear counterexample
is Qwen2.5-3B 4K FWE, which
sits at ratio 0.042 with real headroom (0.24). At N = 100 an
observed rho = 0.01 is a single recovery, Wilson [0.002, 0.054],
so the ratio is dominated by counting noise rather than by the
relationship being modelled.

**Two defensible fixes, pick one and say which:**

1. Replace "slope ~0.4" with the fitted 0.34, and
   exclude the FWE-4K cell from the band with the counting-noise reason
   stated. The band then honestly describes 4 of 5 dilution rows.
2. Keep every row and widen the band to [0.04, 0.55]. This is
   more honest but the band no longer supports "approximately constant",
   so the surrounding prose must weaken accordingly.

Either way, the paper must stop printing three different numbers for one
slope; it is a two-minute recomputation to check.

## Checks

- [PASS] dilution-only slope == 0.3432 (got 0.3432)
- [PASS] with-MK3 slope == 0.2111 (got 0.2111)
- [PASS] fitted slope differs from the caption's 0.4 (got |0.3432 - 0.4| = 0.0568)
- [PASS] FWE-4K falls outside the claimed band (got 2 outside: [0.556, 0.042])
  - outside band: Qwen2.5-3B   16K VT at 0.556
  - outside band: Qwen2.5-3B   4K  FWE at 0.042
