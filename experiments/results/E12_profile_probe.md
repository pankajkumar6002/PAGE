# E12: learned probe over the per-layer profile a_l (Q2 / W4)

Label = the paper's per-input AUC label (heldout_ablation.py): NIAH-MK3 vs the dilution-prone pool (vt, fwe, qa_1, niah_multivalue). Features = interior profile summaries (min, argmin depth, mid-band mean) that the endpoint statistic D discards. Fit on Qwen2.5-1.5B, held-out on Llama-3.1-8B. Pure numpy.

- Fit Qwen2.5-1.5B: N=500 (100 MK3 / 200 dil); corrupt 0.
- Eval Llama-3.1-8B: N=250 (50 MK3 / 200 dil); corrupt 0.

## Held-out AUC on Llama-3.1-8B (NIAH-MK3 vs dilution pool)

| predictor | held-out AUC |
|---|---:|
| raw D | 0.741 |
| z-scored D (paper's current best) | 0.741 |
| **learned profile probe** | **0.571** |

- guard: recomputed drop_D AUC = 1.000 (fit, paper 1.000), 0.741 (eval, paper 0.741 for raw D) — within tol.
- note: z-scoring a single scalar is monotonic, so z-scored-D AUC (0.741) equals raw-D AUC by construction; the paper's ~0.80 refers to z-scored threshold transfer, not per-input AUC on this cell.
- probe in-sample AUC on Qwen2.5-1.5B: 0.986 (D already separates at 1.000 here, so no in-sample headroom).
- probe weights [intercept, min, argmin_depth, mid_band]: [-12.094, 11.531, -9.307, -7.78]

## Verdict

NO IMPROVEMENT: probe 0.571 <= z-scored D 0.741. The endpoint statistic is not the bottleneck on Llama; the failure lives elsewhere.

