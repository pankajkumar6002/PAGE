# Tau sensitivity sweep

Sweep of the gating threshold tau on Qwen2.5-1.5B-Instruct, RULER 4K, mixed suite (NIAH-MK3 + VT + FWE + QA_1), N=100 per task, budgets {0.5, 0.25, 0.125, 0.0625}, SnapKV.

Delta = mean over (task, budget) cells of (gated_acc - plain_acc). FP rate = fraction of capacity-bound MK3 examples on which the gate opens (should be small). FN rate = fraction of dilution-prone (VT, FWE, QA_1) examples on which the gate stays closed (should be small).

| tau | Delta | FP (MK3) | FN (VT+FWE+QA_1) | gated mean acc | plain mean acc |
|---:|---:|---:|---:|---:|---:|
| 0.01 | +0.0000 | 1.000 | 0.000 | 0.368 | 0.368 |
| 0.025 | +0.0025 | 0.990 | 0.000 | 0.371 | 0.368 |
| 0.04 | +0.0550 | 0.640 | 0.000 | 0.423 | 0.368 |
| 0.055 | +0.1431 | 0.090 | 0.000 | 0.511 | 0.368 |
| 0.07 | +0.1525 | 0.000 | 0.000 | 0.521 | 0.368 |
| 0.085 | +0.1525 | 0.000 | 0.000 | 0.521 | 0.368 |
| 0.1 | +0.1525 | 0.000 | 0.000 | 0.521 | 0.368 |
| 0.13 | +0.1688 | 0.000 | 0.333 | 0.537 | 0.368 |
| 0.16 | +0.1869 | 0.000 | 0.443 | 0.555 | 0.368 |
| 0.2 | +0.2238 | 0.000 | 0.673 | 0.592 | 0.368 |

## Plateau analysis

- Delta at deployed tau=0.07: +0.1525
- Unconstrained maximum Delta across the sweep: +0.2238
  (achieved at tau in [0.2, 0.2], where the gate closes on most dilution-prone examples too -- the system effectively reverts to full KV, which is robust by construction but defeats the gating purpose.)
- Operational plateau (|Delta - Delta_0.07| <= 1pp AND FP+FN <= 0.10): tau in [0.055, 0.1]
- Headline: Delta varies by 0.94 pp across tau in [0.055, 0.1]; tau = 0.07 sits inside the plateau.

Figure: paper/figs/tau_sensitivity.pdf (+ .png)
