# Pre-registration: B3 and B4, newer and hybrid models

Written before any B3 or B4 run. Date: 2026-11-07. Answers reviewer items G-W4, G-Q2 and G-Q3.

## B3: one newer, larger model

- **Model.** Qwen3-14B with thinking disabled in the chat template. Fallback if it fails the feasibility check or does not fit: Gemma-4-31B.
- **Feasibility check.** One input: prefill, window attention, cache slicing and a decode all run, and full-cache accuracy is not at floor on a 10-input sample.
- **Design.** RULER 4K, tasks niah_multikey_3, vt, fwe, qa_1, N = 50 per task, SnapKV, budgets {1, 1/2, 1/4, 1/8, 1/16}. Fixed tau = 0.07 and z-gate (theta_z = -0.69, standardized over the cell's inputs).
- **Metrics.** As A1 and A2: AUC(D) for MK3, AUC against harm pooled and within-task, PAGE gain, random gain, selection, achieved cache.

Predictions:
- **P1.** Fixed tau fails to separate on this model, as on Qwen3-4B.
- **P2.** The z-gate closes MK3 on most inputs and shows positive selection.

## B4: a hybrid model

- **Model.** Qwen3.5-9B: 24 linear-attention layers and 8 full-attention layers. A previous run crashed on its linear layers (log not released).
- **Change to the method.** D is computed from the 8 full-attention layers only, as early third minus late third of those layers. Eviction applies to the key-value caches of those layers only. Linear-attention layers keep their state untouched.
- **Feasibility check.** One input: attention is available on full-attention layers, their caches can be sliced with correct positions, and a decode runs and matches the full-cache output at b = 1.
- **Design, if feasible.** As B3.

Predictions:
- **P3.** The feasibility check may fail. If it does, that failure is the reported result, along with how it failed.
- **P4.** If feasible, eviction on full-attention layers alone degrades accuracy less than on a full-attention model at the same nominal budget, because linear layers keep their own summary of the context. This means budgets are not comparable with other models, and it will be stated.

## Reporting

Both are reported as they come out, including failures. MLA and DSA are not tested, because no checkpoint is available here, and the response will say so.
