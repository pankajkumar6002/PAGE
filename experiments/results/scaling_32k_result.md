# Pre-registered 32K scaling-formula test: RESULT (measured after frozen predictions)

Sweep: `ruler_32k_qwen15b_sweep.jsonl` (Qwen2.5-1.5B-Instruct, RULER 32K
via SaylorTwift/RULER-32768-Qwen2.5-3B-tokenizer, N=100/task, 10 budgets,
SnapKV-style eviction, two-pass, greedy, reanalyze_ruler.py correctness).
Predictions were frozen in `scaling_32k_prediction.md`
(sha256 c92966776f754caa71bd5215a209b7f277bd96af7820a3c8f35417aaa9a3422f,
logged 2026-07-03T02:39:33Z) before the sweep launched (02:39:57Z); full
chronology in `scaling_32k_protocol.log`. Consistency check: b=1.0
accuracies in the sweep match the pre-prediction headroom run exactly
(0.810 / 0.490 / 0.140).

## Headline table

| task | A_full (Wilson95) | 1-A_full | measured rho (Wilson95) | predicted rho (PRIMARY, frozen) | verdict | measured ratio rho/(1-A_full) | fitted ratio band |
|---|---|---|---|---|---|---|---|
| VT  | 0.810 [0.722, 0.875] | 0.190 | **0.140** [0.085, 0.221] | [0.048, 0.106] (point 0.072) | **OUTSIDE (above)** | 0.737 | [0.253, 0.556] |
| FWE | 0.490 [0.394, 0.587] | 0.510 | **0.050** [0.022, 0.112] | [0.021, 0.263] (point 0.087) | **INSIDE** | 0.098 | [0.042, 0.516] |
| MK3 | 0.140 [0.085, 0.221] | 0.860 | **0.040** [0.016, 0.098] | rho <= 0.02 (protocol-fixed) | **OUTSIDE (above)** | 0.047 | ~0 (0.000-0.029) |

Score: **1 of 3 primary predictions landed** (FWE). Notes per task:

- **VT**: rho fell from 0.19 (16K) to 0.14 (32K) as headroom collapsed
  (0.75 -> 0.19) — the *direction* the formula dictates — but the measured
  point sits ~32% above the predicted upper bound. The implied ratio 0.737
  exceeds every previously observed dilution-prone cell (max 0.556). The
  predicted interval [0.048, 0.106] does overlap the measured Wilson CI
  [0.085, 0.221], so the miss is not statistically decisive at N=100, but
  by the pre-registered point-estimate rule it is a miss. Striking
  amplification detail: VT accuracy at budget 0.125 (0.850) *exceeds*
  full-KV accuracy (0.810); the budget curve is flat-to-rising as cache
  shrinks — eviction is net-positive on VT at 32K, the strongest dilution
  signature observed in any cell so far.
- **FWE**: 0.050 lands inside the (admittedly wide) primary interval
  [0.021, 0.263]. The sharper pre-registered exploratory model-matched
  band (1.5B-only ratios -> rho in [0.023, 0.039]) is barely exceeded
  (0.050), and the measured ratio 0.098 sits near the 1.5B historical
  ratios (0.045-0.077), far from the 3B-16K ratio (0.516). The FWE ratio
  is model-dependent, not a universal constant.
- **MK3**: 0.040 breaks the rho <= 0.02 pin (4/100 recovered inputs).
  The Wilson CI [0.016, 0.098] still includes 0.02, so at N=100 this is a
  point-estimate miss compatible with the boundary. The capacity-bound
  *contrast* survives strongly in the ratio: MK3 ratio 0.047 despite
  headroom 0.860, vs VT 0.737 at headroom 0.190 — a 15x separation, the
  partition remains visible in the ratio alone. But "rho <= 0.02
  regardless of headroom" as a hard pin is falsified at the point-estimate
  level at 32K.

## Full budget curves (accuracy at each budget, N=100)

| budget | VT | FWE | MK3 |
|---:|---:|---:|---:|
| 1.0    | 0.810 | 0.490 | 0.140 |
| 0.875  | 0.790 | 0.470 | 0.150 |
| 0.75   | 0.770 | 0.440 | 0.150 |
| 0.625  | 0.790 | 0.360 | 0.160 |
| 0.5    | 0.810 | 0.300 | 0.150 |
| 0.375  | 0.820 | 0.200 | 0.120 |
| 0.25   | 0.840 | 0.170 | 0.080 |
| 0.1875 | 0.820 | 0.090 | 0.060 |
| 0.125  | 0.850 | 0.070 | 0.040 |
| 0.0625 | 0.810 | 0.070 | 0.020 |

## Context-amplification check (pre-registered rule 2)

The formula says rho tracks headroom, not context length. Qwen2.5-1.5B
across contexts (4K/16K: simonjegou/ruler; 32K: SaylorTwift RULER — same
generator family, different instantiation, caveat noted):

| task | ctx | 1-A_full | rho | ratio |
|---|---|---:|---:|---:|
| VT  | 4K  | 0.18 | 0.06 | 0.333 |
| VT  | 16K | 0.75 | 0.19 | 0.253 |
| VT  | 32K | 0.19 | 0.14 | 0.737 |
| FWE | 4K  | 0.78 | 0.06 | 0.077 |
| FWE | 16K | 0.88 | 0.04 | 0.045 |
| FWE | 32K | 0.51 | 0.05 | 0.098 |
| MK3 | 4K  | 0.35 | 0.01 | 0.029 |
| MK3 | 16K | 0.74 | 0.00 | 0.000 |
| MK3 | 32K | 0.86 | 0.04 | 0.047 |

- Headroom-tracking direction: PASSES on VT (rho fell 0.19 -> 0.14 when
  headroom fell 0.75 -> 0.19, even though context doubled) and is
  consistent on FWE (rho ~flat as ratio stays low).
- Proportionality-constant stability: FAILS on VT — the ratio roughly
  triples at 32K (0.25-0.33 -> 0.74). Recovery per unit headroom GROWS
  with context on the dilution-prone task; the paper's constant-ratio
  band [0.25, 0.55] undersells dilution at 32K. The strengthened form
  (eviction beats full KV outright at b <= 0.25 on VT) is consistent with
  the dilution mechanism the formula is meant to capture, i.e., the miss
  is in the formula's conservative direction.
- Capacity-bound side: MK3 ratio stays an order of magnitude below the
  dilution-prone task's ratio at every context; the hard rho <= 0.02 pin
  slips to 0.04 at 32K (CI-compatible with 0.02).

## Not measured

Mean head-agreement drop D per task at 32K: skipped. ruler_sweep.py does
not compute drops, and the protocol forbids modifying it; the drop probe
(head_agreement_probe.py) would be a separate run.

## Provenance

- Fit (existing data only): logged 02:15:58Z, all 8 audited paper-table
  cells reproduced exactly.
- Headroom run: `ruler_32k_qwen15b_afull.jsonl` (b=1.0 only), done
  02:39Z, sha256 5b30ad27ab3a1cdaf58035e81271f5311ce9feeb071a140a47973898ae9ed4e1.
- Predictions frozen + hashed 02:39:33Z, BEFORE sweep launch 02:39:57Z.
- Sweep: 3000 records, wall time 2726s, done ~03:26Z, GPU 1.
- Wilson intervals: two-sided 95%, score method.
