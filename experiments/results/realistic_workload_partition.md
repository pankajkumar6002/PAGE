# Realistic-workload partition + gate test (Qwen2.5-14B-Instruct, LongBench)

- tau = 0.07; budgets [1.0, 0.5, 0.25, 0.125, 0.0625]; two_pass; obs_window 32, n_sink 4, top_k 32.
- Plain = SnapKV-style eviction always on. Gated = evict only if gate open (D>=tau).
- Metric: any-in substring match (gold answer/line appears in prediction).

## Summary table

| Task | N | T_med | A_full | acc@0.5 | acc@0.125 | acc@0.0625 | full-correct retained @0.0625 | rho | mean D | gate-open frac | gate predicts | empirical class | gate correct? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hotpotqa | 100 | 15180 | 0.560 | 0.560 | 0.540 | 0.490 | 49/56 (0.88) | 0.020 | +0.0973 | 0.860 | evict (dilution/robust) | evict-robust | YES |

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
- **Empirical class: evict-robust**; gate predicts: evict (dilution/robust); gate correct: YES

