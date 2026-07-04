# CapKV vs gated-CapKV on LongBench/qasper
## Paper-ready summary (for main.tex citation)
- N = **68** examples (LongBench/qasper, Qwen2.5-3B-Instruct, budget b=**0.50**, CapKV temperature tau=5.0).
- plain CapKV accuracy (any-in substring match) = **0.1324**
- gated CapKV accuracy (any-in substring match) = **0.1471**
- **Δ (gated − plain) = +0.0147** (+1/68 examples flipped)
- gate-open rate = 45/68 = 0.662

## Experimental setup
- Model: Qwen2.5-3B-Instruct (eager attention, bf16)
- Subtask: LongBench/qasper
- N = 68 examples (skip if tokenized prompt > 16K tokens)
- Median prompt length T = 4538 tokens
- Eviction budget b = 0.50 (median kept tokens: plain 2269, gated 2646)
- CapKV proxy: per-layer leverage score s_i = w_i * v_i^T A^{-1} v_i, with A = I + sum_i w_i v_i v_i^T and w_i = exp(<k_i, mu_q> * tau / sqrt(d)). mu_q = mean q over the last 32 prompt positions. tau = 5.0.
- Gating wrapper: keep full KV when head-agreement early-vs-late drop < 0.07; otherwise evict at budget b using CapKV-proxy score.
- Metric: any-in substring match against the gold answer list.

## Headline
| variant | acc | n_correct |
|---|---:|---:|
| plain CapKV (b=0.5) | 0.1324 | 9/68 |
| gated CapKV (b=0.5) | 0.1471 | 10/68 |
| **Δ (gated − plain)** | **+0.0147** | +1 |

## Conditional breakdown
| subset | n | plain acc | gated acc | Δ |
|---|---:|---:|---:|---:|
| gate-open  (drop ≥ 0.07) | 45 | 0.1556 | 0.1556 | +0.0000 |
| gate-closed (drop < 0.07) | 23 | 0.0870 | 0.1304 | +0.0435 |

Gate-open Δ is exactly 0 by construction: when the gate fires, both plain and gated apply the same CapKV-proxy eviction at b=0.5. Gate-closed Δ shows the gating mechanism's lift: gated keeps the full KV (do_evict=False) while plain still evicts at b=0.5, and on the examples the gate flags as capacity-bound this avoids the plain-CapKV accuracy hit.
