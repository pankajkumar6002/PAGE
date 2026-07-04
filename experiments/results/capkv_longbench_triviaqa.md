# CapKV vs gated-CapKV on LongBench/LongBench/triviaqa
## Paper-ready summary (for main.tex citation)
- N = **24** examples (LongBench/LongBench/triviaqa, Qwen2.5-3B-Instruct, budget b=**0.50**, CapKV temperature tau=5.0).
- plain CapKV accuracy (any-in substring match) = **0.8750**
- gated CapKV accuracy (any-in substring match) = **0.9583**
- **Δ (gated − plain) = +0.0833** (+2/24 examples flipped)
- gate-open rate = 13/24 = 0.542

## Experimental setup
- Model: Qwen2.5-3B-Instruct (eager attention, bf16)
- Subtask: LongBench/LongBench/triviaqa
- N = 24 examples (skip if tokenized prompt > 16K tokens)
- Median prompt length T = 4793 tokens
- Eviction budget b = 0.50 (median kept tokens: plain 2396, gated 2893)
- CapKV proxy: per-layer leverage score s_i = w_i * v_i^T A^{-1} v_i, with A = I + sum_i w_i v_i v_i^T and w_i = exp(<k_i, mu_q> * tau / sqrt(d)). mu_q = mean q over the last 32 prompt positions. tau = 5.0.
- Gating wrapper: keep full KV when head-agreement early-vs-late drop < 0.07; otherwise evict at budget b using CapKV-proxy score.
- Metric: any-in substring match against the gold answer list.

## Headline
| variant | acc | n_correct |
|---|---:|---:|
| plain CapKV (b=0.5) | 0.8750 | 21/24 |
| gated CapKV (b=0.5) | 0.9583 | 23/24 |
| **Δ (gated − plain)** | **+0.0833** | +2 |

## Conditional breakdown
| subset | n | plain acc | gated acc | Δ |
|---|---:|---:|---:|---:|
| gate-open  (drop ≥ 0.07) | 13 | 1.0000 | 1.0000 | +0.0000 |
| gate-closed (drop < 0.07) | 11 | 0.7273 | 0.9091 | +0.1818 |

Gate-open Δ is exactly 0 by construction: when the gate fires, both plain and gated apply the same CapKV-proxy eviction at b=0.5. Gate-closed Δ shows the gating mechanism's lift: gated keeps the full KV (do_evict=False) while plain still evicts at b=0.5, and on the examples the gate flags as capacity-bound this avoids the plain-CapKV accuracy hit.
