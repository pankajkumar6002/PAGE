# P3-1: seed variance of the gating $\Delta$

Input-draw seeds (1, 2, 3), greedy decoding, $\tau = 0.07$, budgets $b < 1.0$. Each cell's seeds share a host.

| cell | seed 1 | seed 2 | seed 3 | mean | SD | range |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B | +0.113 | +0.099 | +0.122 | +0.111 | 0.011 | 0.022 |
| Qwen2.5-3B | +0.255 | +0.267 | +0.275 | +0.266 | 0.010 | 0.019 |
| Qwen2.5-14B | +0.133 | +0.140 | +0.142 | +0.138 | 0.005 | 0.010 |
| Mistral-7B | +0.187 | +0.194 | +0.183 | +0.188 | 0.006 | 0.011 |
| Llama-3.1-8B | +0.078 | +0.071 | +0.078 | +0.076 | 0.004 | 0.007 |

Restricted to NIAH-MK3, where almost all of $\Delta$ lives:

| cell | seed 1 | seed 2 | seed 3 | mean | SD | range |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B | +0.454 | +0.398 | +0.486 | +0.446 | 0.045 | 0.089 |
| Qwen2.5-3B | +0.738 | +0.747 | +0.729 | +0.738 | 0.009 | 0.019 |
| Qwen2.5-14B | +0.530 | +0.559 | +0.568 | +0.552 | 0.020 | 0.037 |
| Mistral-7B | +0.709 | +0.710 | +0.669 | +0.696 | 0.023 | 0.041 |
| Llama-3.1-8B | +0.310 | +0.282 | +0.311 | +0.301 | 0.016 | 0.029 |

MK3-only seed SD is 0.009 to 0.045, on effects of +0.301 to +0.738.

Across the 5 complete cells the seed SD is 0.004 to 0.011, against a grand-mean $\Delta$ of $+0.229$pp in Table~\ref{tab:matrix}. The headline is therefore not an artifact of one input draw: draw-to-draw spread is roughly an order of magnitude below the effect it is measuring.
