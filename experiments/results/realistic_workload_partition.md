# Realistic-workload partition + gate test (Qwen2.5-14B-Instruct, LongBench)

- tau = 0.07; budgets [1.0, 0.5, 0.25, 0.125, 0.0625]; two_pass; obs_window 32, n_sink 4, top_k 32.
- Plain = SnapKV-style eviction always on. Gated = evict only if gate open (D>=tau).
- Metric: any-in substring match (gold answer/line appears in prediction).

## Summary table

| Task | N | T_med | A_full | acc@0.5 | acc@0.125 | acc@0.0625 | full-correct retained @0.0625 | rho | mean D | gate-open frac | gate predicts | empirical class | gate correct? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lcc | 27 | 2431 | 0.296 | 0.296 | 0.222 | 0.185 | 3/8 (0.38) | 0.074 | +0.0580 | 0.296 | do-not-evict (capacity-bound) | dilution-prone | NO |
| repobench-p | 18 | 12115 | 0.333 | 0.333 | 0.444 | 0.278 | 3/6 (0.50) | 0.167 | +0.0454 | 0.111 | do-not-evict (capacity-bound) | dilution-prone | NO |

## lcc  (N=27, T 1355-6020, median 2431)

- A_full = 0.296  (8 full-KV-correct inputs)

| budget | plain acc | gated acc | full-correct retained (plain) |
|---:|---:|---:|---:|
| 1.0 | 0.296 | 0.296 | 8/8 |
| 0.5 | 0.296 | 0.296 | 7/8 |
| 0.25 | 0.259 | 0.296 | 6/8 |
| 0.125 | 0.222 | 0.296 | 5/8 |
| 0.0625 | 0.185 | 0.296 | 3/8 |

- rho = 0.074   (Pr[full-KV wrong AND some b<1.0 correct])
- mean head-agreement drop D = +0.0580 (std 0.0217); gate-open fraction at tau=0.07: 0.296
- accuracy loss full->b_min: 38%; retention of full-correct at b_min: 0.38
- **Empirical class: dilution-prone**; gate predicts: do-not-evict (capacity-bound); gate correct: NO

## repobench-p  (N=18, T 3294-19927, median 12115)

- A_full = 0.333  (6 full-KV-correct inputs)

| budget | plain acc | gated acc | full-correct retained (plain) |
|---:|---:|---:|---:|
| 1.0 | 0.333 | 0.333 | 6/6 |
| 0.5 | 0.333 | 0.333 | 6/6 |
| 0.25 | 0.444 | 0.333 | 6/6 |
| 0.125 | 0.444 | 0.333 | 5/6 |
| 0.0625 | 0.278 | 0.278 | 3/6 |

- rho = 0.167   (Pr[full-KV wrong AND some b<1.0 correct])
- mean head-agreement drop D = +0.0454 (std 0.0216); gate-open fraction at tau=0.07: 0.111
- accuracy loss full->b_min: 17%; retention of full-correct at b_min: 0.50
- **Empirical class: dilution-prone**; gate predicts: do-not-evict (capacity-bound); gate correct: NO

