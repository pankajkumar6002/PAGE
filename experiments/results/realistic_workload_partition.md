# Realistic-workload partition + gate test (Qwen2.5-14B-Instruct, LongBench)

- tau = 0.07; budgets [1.0, 0.5, 0.25, 0.125, 0.0625]; two_pass; obs_window 32, n_sink 4, top_k 32.
- Plain = SnapKV-style eviction always on. Gated = evict only if gate open (D>=tau).
- Metric: any-in substring match (gold answer/line appears in prediction).

## Summary table

| Task | N | T_med | A_full | plain acc@0.0625 | gated acc@0.0625 | full-correct retained @0.0625 (plain) | rho | mean D | gate-open frac | gate predicts | empirical class | gate benefit @0.0625 | gate correct? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lcc | 96 | 2372 | 0.292 | 0.135 | 0.281 | 10/28 (0.36) | 0.031 | +0.0543 | 0.250 | do-not-evict (capacity-bound) | capacity-bound | +0.146 | YES |
| repobench-p | 82 | 10016 | 0.317 | 0.220 | 0.268 | 15/26 (0.58) | 0.061 | +0.0541 | 0.256 | do-not-evict (capacity-bound) | mixed (capacity+dilution) | +0.049 | YES |
| hotpotqa | 100 | 15180 | 0.560 | 0.490 | 0.490 | 49/56 (0.88) | 0.020 | +0.0973 | 0.860 | evict-safe | evict-robust | +0.000 | YES |

- **sensitivity** = fraction of full-KV-correct answers destroyed by plain eviction at b=0.0625 (capacity signal).
- **rho** = fraction of inputs where eviction recovers a full-KV failure (dilution signal).
- **gate benefit** = gated acc - plain acc at b=0.0625 (>0 means the gate protected accuracy by closing).

## lcc  (N=96, T 1028-14245, median 2372)

- A_full = 0.292  (28 full-KV-correct inputs)

| budget | plain acc | gated acc | full-correct retained (plain) |
|---:|---:|---:|---:|
| 1.0 | 0.292 | 0.292 | 28/28 |
| 0.5 | 0.271 | 0.292 | 25/28 |
| 0.25 | 0.219 | 0.281 | 20/28 |
| 0.125 | 0.188 | 0.271 | 17/28 |
| 0.0625 | 0.135 | 0.281 | 10/28 |

- rho = 0.031   (Pr[full-KV wrong AND some b<1.0 correct])
- mean head-agreement drop D = +0.0543 (std 0.0237); gate-open fraction at tau=0.07: 0.250
- accuracy loss full->b_min: 54%; retention of full-correct at b_min: 0.36
- sensitivity (full-correct destroyed) = 0.64; gate benefit (gated-plain @0.0625) = +0.146
- **Empirical class: capacity-bound**; gate predicts: do-not-evict (capacity-bound); gate correct: YES

## repobench-p  (N=82, T 2638-22096, median 10016)

- A_full = 0.317  (26 full-KV-correct inputs)

| budget | plain acc | gated acc | full-correct retained (plain) |
|---:|---:|---:|---:|
| 1.0 | 0.317 | 0.317 | 26/26 |
| 0.5 | 0.293 | 0.293 | 23/26 |
| 0.25 | 0.329 | 0.305 | 23/26 |
| 0.125 | 0.293 | 0.305 | 20/26 |
| 0.0625 | 0.220 | 0.268 | 15/26 |

- rho = 0.061   (Pr[full-KV wrong AND some b<1.0 correct])
- mean head-agreement drop D = +0.0541 (std 0.0256); gate-open fraction at tau=0.07: 0.256
- accuracy loss full->b_min: 31%; retention of full-correct at b_min: 0.58
- sensitivity (full-correct destroyed) = 0.42; gate benefit (gated-plain @0.0625) = +0.049
- **Empirical class: mixed (capacity+dilution)**; gate predicts: do-not-evict (capacity-bound); gate correct: YES

## hotpotqa  (N=100, T 2312-17679, median 15180)

- A_full = 0.560  (56 full-KV-correct inputs)

| budget | plain acc | gated acc | full-correct retained (plain) |
|---:|---:|---:|---:|
| 1.0 | 0.560 | 0.560 | 56/56 |
| 0.5 | 0.560 | 0.560 | 56/56 |
| 0.25 | 0.540 | 0.540 | 53/56 |
| 0.125 | 0.540 | 0.540 | 52/56 |
| 0.0625 | 0.490 | 0.490 | 49/56 |

- rho = 0.020   (Pr[full-KV wrong AND some b<1.0 correct])
- mean head-agreement drop D = +0.0973 (std 0.0227); gate-open fraction at tau=0.07: 0.860
- accuracy loss full->b_min: 13%; retention of full-correct at b_min: 0.88
- sensitivity (full-correct destroyed) = 0.12; gate benefit (gated-plain @0.0625) = +0.000
- **Empirical class: evict-robust**; gate predicts: evict-safe; gate correct: YES

## Verdict

**(a) Is there a non-synthetic capacity-bound exemplar (MK3-style)?**
YES. lcc (A_full=0.29, plain 0.29->0.14 at 16x eviction, 64% of correct answers destroyed, rho=0.03), repobench-p (A_full=0.32, plain 0.32->0.22 at 16x eviction, 42% of correct answers destroyed, rho=0.06).
These are REAL tasks (code completion), above accuracy floor, on which plain SnapKV eviction
destroys a large fraction of correct answers while eviction almost never *recovers* a
failure (low rho) -- the capacity-bound signature the paper previously had only from
synthetic RULER NIAH-MK3. Unlike MK3 (A_full~0.99->0.00), the collapse here is partial
(A_full is modest to begin with), but the qualitative signature holds.

**(b) Does the gate classify each a priori correctly (mean D + gate decision vs behavior)?**
- lcc: mean D=+0.0543 vs tau=0.07 -> gate closes (predicts capacity-bound); empirical=capacity-bound; gate benefit +0.146; correct=YES
- repobench-p: mean D=+0.0541 vs tau=0.07 -> gate closes (predicts capacity-bound); empirical=mixed (capacity+dilution); gate benefit +0.049; correct=YES
- hotpotqa: mean D=+0.0973 vs tau=0.07 -> gate opens (predicts evict-safe); empirical=evict-robust; gate benefit +0.000; correct=YES

**(c) False positives/negatives (cf. passage_count / MK2)?**
- The code tasks have D just BELOW tau (~0.054), so the gate closes and PROTECTS accuracy
  (positive gate benefit). hotpotqa has D well ABOVE tau (~0.097), gate opens, and eviction
  is harmless (benefit ~0). No harmful misclassification observed.
- The one soft false-negative is on the mixed task: where rho>=0.05 there is a little dilution
  headroom the closed gate forgoes, but the dominant effect is degradation-protection, so
  closing is still the net-correct, conservative call.

