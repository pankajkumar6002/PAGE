
## Newer model, Qwen3-14B with thinking off, RULER 4K suite

Qwen3-14B: 200 inputs (fwe 50, niah_multikey_3 50, qa_1 50, vt 50), budgets 1/16, 1/8, 1/4, 1/2, inputs dropped for missing budgets 0, unreadable lines 0.
AUC of D for NIAH-MK3 (MK3 low) against the other three tasks: 0.922. Mean D by task: fwe +0.043, niah_multikey_3 -0.003, qa_1 +0.060, vt +0.009.

### Table 43: Qwen3-14B: gate gain over plain SnapKV, averaged over the budget grid

| gate | open | MK3 closed | other tasks closed | mean kept | full | plain | gated | PAGE gain | random gain | selection | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tau=0.07 | 0.030 | 1.000 | 0.960 | 0.977 | 0.955 | 0.532 | 0.956 | +0.424 | +0.410 | +0.014 | [+0.004, +0.025] |
| z-gate (0.009) | 0.660 | 0.840 | 0.173 | 0.495 | 0.955 | 0.532 | 0.751 | +0.219 | +0.144 | +0.075 | [+0.059, +0.090] |

Random gain = (1 - open) x (full - plain): a gate that opens at random at the same rate. Selection = PAGE gain - random gain. Bootstrap over inputs within task, 2000 replicates. The z-gate threshold is mean D + (-0.69) SD over all 200 inputs.

### Qwen3-14B: accuracy by task and budget, plain SnapKV

| task | N | mean D | full | 1/16 | 1/8 | 1/4 | 1/2 | z-gate, grid mean |
|---|---|---|---|---|---|---|---|---|
| fwe | 50 | +0.043 | 0.940 | 0.020 | 0.360 | 0.660 | 0.780 | 0.455 |
| niah_multikey_3 | 50 | -0.003 | 1.000 | 0.000 | 0.000 | 0.020 | 0.500 | 0.865 |
| qa_1 | 50 | +0.060 | 0.880 | 0.680 | 0.860 | 0.900 | 0.900 | 0.835 |
| vt | 50 | +0.009 | 1.000 | 0.000 | 0.840 | 1.000 | 1.000 | 0.850 |

### Qwen3-14B: D against per-input harm, with the comparators of Table 33

| model | b | full correct | harmed | pooled AUC(D) | 95% CI | task id (in-sample) | task id (split-half) | D (same splits) | within-task AUC | 95% CI | same-task pairs | per task (at least 5 in each class) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-14B | 1/16 | 191 | 157 | 0.956 | [0.925, 0.981] | 0.960 | 0.953 | 0.958 | 0.631 | [0.432, 0.816] | 409 | qa_1 0.59 |
| Qwen3-14B | 1/4 | 191 | 63 | 0.809 | [0.760, 0.859] | 0.967 | 0.968 | 0.806 | 0.538 | [0.353, 0.703] | 511 | fwe 0.51 |

Harm = correct at full cache and wrong at b. AUC of -D, so above 0.5 is the direction the gate assumes.
