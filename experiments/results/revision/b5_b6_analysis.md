
## B5. Hybrid models, D and eviction on the layers with a key-value cache (G-Q3, G-Q2)


### B5a. Olmo-Hybrid-7B (linear + full attention)

Olmo-Hybrid-7B: 200 inputs (fwe 50, niah_multikey_3 50, qa_1 50, vt 50), budgets 1/16, 1/8, 1/4, 1/2, inputs dropped for missing budgets 0, unreadable lines 0.
AUC of D for NIAH-MK3 (MK3 low) against the other three tasks: 1.000. Mean D by task: fwe -0.016, niah_multikey_3 -0.215, qa_1 +0.010, vt +0.002.

### B5a-T1: Olmo-Hybrid-7B: gate gain over plain SnapKV, averaged over the budget grid

| gate | open | MK3 closed | other tasks closed | mean kept | full | plain | gated | PAGE gain | random gain | selection | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tau=0.07 | 0.000 | 1.000 | 1.000 | 1.000 | 0.905 | 0.598 | 0.905 | +0.307 | +0.307 | +0.000 | [+0.000, +0.000] |
| z-gate (-0.121) | 0.750 | 1.000 | 0.000 | 0.426 | 0.905 | 0.598 | 0.776 | +0.179 | +0.077 | +0.102 | [+0.090, +0.113] |

Random gain = (1 - open) x (full - plain): a gate that opens at random at the same rate. Selection = PAGE gain - random gain. Bootstrap over inputs within task, 2000 replicates. The z-gate threshold is mean D + (-0.69) SD over all 200 inputs.

### B5a-T2: Olmo-Hybrid-7B: accuracy by task and budget, plain SnapKV

| task | N | mean D | full | 1/16 | 1/8 | 1/4 | 1/2 | z-gate, grid mean |
|---|---|---|---|---|---|---|---|---|
| fwe | 50 | -0.016 | 0.860 | 0.280 | 0.540 | 0.720 | 0.820 | 0.590 |
| niah_multikey_3 | 50 | -0.215 | 0.980 | 0.000 | 0.020 | 0.160 | 0.880 | 0.980 |
| qa_1 | 50 | +0.010 | 0.880 | 0.760 | 0.840 | 0.920 | 0.880 | 0.850 |
| vt | 50 | +0.002 | 0.900 | 0.000 | 0.940 | 0.900 | 0.900 | 0.685 |

### B5a-T3: Olmo-Hybrid-7B: D against per-input harm, with A1-T1's comparators

| model | b | full correct | harmed | pooled AUC(D) | 95% CI | task id (in-sample) | task id (split-half) | D (same splits) | within-task AUC | 95% CI | same-task pairs | per task (at least 5 in each class) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Olmo-Hybrid-7B | 1/16 | 181 | 131 | 0.703 | [0.635, 0.773] | 0.937 | 0.934 | 0.707 | 0.390 | [0.258, 0.519] | 649 | fwe 0.39, qa_1 0.38 |
| Olmo-Hybrid-7B | 1/4 | 181 | 49 | 0.920 | [0.867, 0.966] | 0.943 | 0.943 | 0.920 | 0.541 | [0.391, 0.691] | 608 | fwe 0.39, niah_multikey_3 0.67 |

Harm = correct at full cache and wrong at b. AUC of -D, so above 0.5 is the direction the gate assumes.

### B5a-T4: Olmo-Hybrid-7B: selection against a random gate at each budget

| b | z-gate selection | 95% CI | tau=0.07 selection | 95% CI |
|---|---|---|---|---|
| 1/16 | +0.084 | [+0.070, +0.098] | +0.000 | [+0.000, +0.000] |
| 1/8 | +0.160 | [+0.145, +0.174] | +0.000 | [+0.000, +0.000] |
| 1/4 | +0.147 | [+0.124, +0.167] | +0.000 | [+0.000, +0.000] |
| 1/2 | +0.016 | [+0.003, +0.033] | +0.000 | [+0.000, +0.000] |

Bootstrap over inputs within task, 2000 replicates.

### B5b. Qwen3-Next-80B-A3B (linear + full attention)

Qwen3-Next-80B-A3B: 200 inputs (fwe 50, niah_multikey_3 50, qa_1 50, vt 50), budgets 1/16, 1/8, 1/4, 1/2, inputs dropped for missing budgets 0, unreadable lines 0.
AUC of D for NIAH-MK3 (MK3 low) against the other three tasks: 1.000. Mean D by task: fwe -0.006, niah_multikey_3 -0.077, qa_1 -0.014, vt -0.029.

### B5b-T1: Qwen3-Next-80B-A3B: gate gain over plain SnapKV, averaged over the budget grid

| gate | open | MK3 closed | other tasks closed | mean kept | full | plain | gated | PAGE gain | random gain | selection | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tau=0.07 | 0.000 | 1.000 | 1.000 | 1.000 | 0.980 | 0.575 | 0.980 | +0.405 | +0.405 | +0.000 | [+0.000, +0.000] |
| z-gate (-0.052) | 0.760 | 0.940 | 0.007 | 0.418 | 0.980 | 0.575 | 0.740 | +0.165 | +0.097 | +0.068 | [+0.056, +0.079] |

Random gain = (1 - open) x (full - plain): a gate that opens at random at the same rate. Selection = PAGE gain - random gain. Bootstrap over inputs within task, 2000 replicates. The z-gate threshold is mean D + (-0.69) SD over all 200 inputs.

### B5b-T2: Qwen3-Next-80B-A3B: accuracy by task and budget, plain SnapKV

| task | N | mean D | full | 1/16 | 1/8 | 1/4 | 1/2 | z-gate, grid mean |
|---|---|---|---|---|---|---|---|---|
| fwe | 50 | -0.006 | 0.980 | 0.240 | 0.420 | 0.620 | 0.780 | 0.515 |
| niah_multikey_3 | 50 | -0.077 | 1.000 | 0.000 | 0.000 | 0.240 | 0.940 | 0.955 |
| qa_1 | 50 | -0.014 | 0.940 | 0.840 | 0.920 | 0.940 | 0.940 | 0.910 |
| vt | 50 | -0.029 | 1.000 | 0.000 | 0.320 | 1.000 | 1.000 | 0.580 |

### B5b-T3: Qwen3-Next-80B-A3B: D against per-input harm, with A1-T1's comparators

| model | b | full correct | harmed | pooled AUC(D) | 95% CI | task id (in-sample) | task id (split-half) | D (same splits) | within-task AUC | 95% CI | same-task pairs | per task (at least 5 in each class) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-Next-80B-A3B | 1/16 | 196 | 143 | 0.746 | [0.685, 0.804] | 0.945 | 0.945 | 0.749 | 0.401 | [0.248, 0.564] | 690 | fwe 0.36, qa_1 0.47 |
| Qwen3-Next-80B-A3B | 1/4 | 196 | 57 | 0.704 | [0.624, 0.784] | 0.906 | 0.910 | 0.705 | 0.452 | [0.329, 0.574] | 1026 | fwe 0.40, niah_multikey_3 0.52 |

Harm = correct at full cache and wrong at b. AUC of -D, so above 0.5 is the direction the gate assumes.

### B5b-T4: Qwen3-Next-80B-A3B: selection against a random gate at each budget

| b | z-gate selection | 95% CI | tau=0.07 selection | 95% CI |
|---|---|---|---|---|
| 1/16 | +0.065 | [+0.051, +0.076] | +0.000 | [+0.000, +0.000] |
| 1/8 | +0.099 | [+0.085, +0.115] | +0.000 | [+0.000, +0.000] |
| 1/4 | +0.108 | [+0.083, +0.133] | +0.000 | [+0.000, +0.000] |
| 1/2 | -0.001 | [-0.013, +0.014] | +0.000 | [+0.000, +0.000] |

Bootstrap over inputs within task, 2000 replicates.

### B5c. LFM2-24B-A2B (short convolution + attention)

LFM2-24B-A2B: 200 inputs (fwe 50, niah_multikey_3 50, qa_1 50, vt 50), budgets 1/16, 1/8, 1/4, 1/2, inputs dropped for missing budgets 0, unreadable lines 0.
AUC of D for NIAH-MK3 (MK3 low) against the other three tasks: 0.946. Mean D by task: fwe -0.015, niah_multikey_3 -0.047, qa_1 -0.031, vt +0.020.

### B5c-T1: LFM2-24B-A2B: gate gain over plain SnapKV, averaged over the budget grid

| gate | open | MK3 closed | other tasks closed | mean kept | full | plain | gated | PAGE gain | random gain | selection | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tau=0.07 | 0.000 | 1.000 | 1.000 | 1.000 | 0.505 | 0.274 | 0.505 | +0.231 | +0.231 | +0.000 | [+0.000, +0.000] |
| z-gate (-0.037) | 0.675 | 1.000 | 0.100 | 0.483 | 0.505 | 0.274 | 0.436 | +0.163 | +0.075 | +0.087 | [+0.070, +0.103] |

Random gain = (1 - open) x (full - plain): a gate that opens at random at the same rate. Selection = PAGE gain - random gain. Bootstrap over inputs within task, 2000 replicates. The z-gate threshold is mean D + (-0.69) SD over all 200 inputs.

### B5c-T2: LFM2-24B-A2B: accuracy by task and budget, plain SnapKV

| task | N | mean D | full | 1/16 | 1/8 | 1/4 | 1/2 | z-gate, grid mean |
|---|---|---|---|---|---|---|---|---|
| fwe | 50 | -0.015 | 0.300 | 0.000 | 0.020 | 0.100 | 0.220 | 0.080 |
| niah_multikey_3 | 50 | -0.047 | 1.000 | 0.000 | 0.060 | 0.520 | 0.860 | 1.000 |
| qa_1 | 50 | -0.031 | 0.720 | 0.500 | 0.740 | 0.680 | 0.680 | 0.665 |
| vt | 50 | +0.020 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

### B5c-T3: LFM2-24B-A2B: D against per-input harm, with A1-T1's comparators

| model | b | full correct | harmed | pooled AUC(D) | 95% CI | task id (in-sample) | task id (split-half) | D (same splits) | within-task AUC | 95% CI | same-task pairs | per task (at least 5 in each class) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LFM2-24B-A2B | 1/16 | 101 | 77 | 0.765 | [0.691, 0.837] | 0.922 | 0.920 | 0.762 | 0.618 | [0.402, 0.807] | 288 | qa_1 0.62 |
| LFM2-24B-A2B | 1/4 | 101 | 39 | 0.501 | [0.390, 0.618] | 0.777 | 0.771 | 0.494 | 0.428 | [0.293, 0.580] | 759 | niah_multikey_3 0.38 |

Harm = correct at full cache and wrong at b. AUC of -D, so above 0.5 is the direction the gate assumes.

### B5c-T4: LFM2-24B-A2B: selection against a random gate at each budget

| b | z-gate selection | 95% CI | tau=0.07 selection | 95% CI |
|---|---|---|---|---|
| 1/16 | +0.147 | [+0.128, +0.165] | +0.000 | [+0.000, +0.000] |
| 1/8 | +0.133 | [+0.111, +0.152] | +0.000 | [+0.000, +0.000] |
| 1/4 | +0.062 | [+0.034, +0.090] | +0.000 | [+0.000, +0.000] |
| 1/2 | +0.009 | [-0.013, +0.032] | +0.000 | [+0.000, +0.000] |

Bootstrap over inputs within task, 2000 replicates.

### B5-T0: B5 summary: D, gates, selection at b = 1/16, and accuracy kept by plain SnapKV

| model | design | N | full accuracy by task | AUC(D) for MK3 | tau open | tau MK3 closed | z open | z MK3 closed | z selection at 1/16 | 95% CI | kept 1/2 | kept 1/4 | kept 1/8 | kept 1/16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Olmo-Hybrid-7B | linear + full attention | 200 | fwe 0.86, niah_multikey_3 0.98, qa_1 0.88, vt 0.90 | 1.000 | 0.000 | 1.000 | 0.750 | 1.000 | +0.084 | [+0.070, +0.098] | 0.961 | 0.746 | 0.646 | 0.287 |
| Qwen3-Next-80B-A3B | linear + full attention | 200 | fwe 0.98, niah_multikey_3 1.00, qa_1 0.94, vt 1.00 | 1.000 | 0.000 | 1.000 | 0.760 | 0.940 | +0.065 | [+0.051, +0.076] | 0.934 | 0.714 | 0.423 | 0.276 |
| LFM2-24B-A2B | short convolution + attention | 200 | fwe 0.30, niah_multikey_3 1.00, qa_1 0.72, vt 0.00 (below 0.2: harm numbers rest on few inputs) | 0.946 | 0.000 | 1.000 | 0.675 | 1.000 | +0.147 | [+0.128, +0.165] | 0.871 | 0.644 | 0.406 | 0.248 |

Kept = plain SnapKV accuracy at b over full-cache accuracy, all four tasks. Eviction applies only to layers with a key-value cache, so for hybrids nominal budgets are not comparable with full-attention models. Kept fractions use T - 1 positions (last prompt token held back).

## B6. MLA models (G-Q3)


### B6a. DeepSeek-V2-Lite-Chat (MLA)

DeepSeek-V2-Lite-Chat: 200 inputs (fwe 50, niah_multikey_3 50, qa_1 50, vt 50), budgets 1/16, 1/8, 1/4, 1/2, inputs dropped for missing budgets 0, unreadable lines 0.
AUC of D for NIAH-MK3 (MK3 low) against the other three tasks: 1.000. Mean D by task: fwe +0.042, niah_multikey_3 -0.049, qa_1 +0.035, vt +0.015.

### B6a-T1: DeepSeek-V2-Lite-Chat: gate gain over plain SnapKV, averaged over the budget grid

| gate | open | MK3 closed | other tasks closed | mean kept | full | plain | gated | PAGE gain | random gain | selection | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tau=0.07 | 0.000 | 1.000 | 1.000 | 1.000 | 0.590 | 0.302 | 0.590 | +0.287 | +0.287 | +0.000 | [+0.000, +0.000] |
| z-gate (-0.015) | 0.750 | 1.000 | 0.000 | 0.426 | 0.590 | 0.302 | 0.455 | +0.153 | +0.072 | +0.081 | [+0.054, +0.107] |

Random gain = (1 - open) x (full - plain): a gate that opens at random at the same rate. Selection = PAGE gain - random gain. Bootstrap over inputs within task, 2000 replicates. The z-gate threshold is mean D + (-0.69) SD over all 200 inputs.

### B6a-T2: DeepSeek-V2-Lite-Chat: accuracy by task and budget, plain SnapKV

| task | N | mean D | full | 1/16 | 1/8 | 1/4 | 1/2 | z-gate, grid mean |
|---|---|---|---|---|---|---|---|---|
| fwe | 50 | +0.042 | 0.420 | 0.020 | 0.160 | 0.160 | 0.300 | 0.160 |
| niah_multikey_3 | 50 | -0.049 | 0.620 | 0.000 | 0.000 | 0.000 | 0.040 | 0.620 |
| qa_1 | 50 | +0.035 | 0.800 | 0.700 | 0.720 | 0.780 | 0.820 | 0.755 |
| vt | 50 | +0.015 | 0.520 | 0.000 | 0.300 | 0.420 | 0.420 | 0.285 |

### B6a-T3: DeepSeek-V2-Lite-Chat: D against per-input harm, with A1-T1's comparators

| model | b | full correct | harmed | pooled AUC(D) | 95% CI | task id (in-sample) | task id (split-half) | D (same splits) | within-task AUC | 95% CI | same-task pairs | per task (at least 5 in each class) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DeepSeek-V2-Lite-Chat | 1/16 | 118 | 86 | 0.741 | [0.659, 0.821] | 0.953 | 0.954 | 0.739 | 0.492 | [0.258, 0.731] | 256 | qa_1 0.49 |
| DeepSeek-V2-Lite-Chat | 1/4 | 118 | 56 | 0.777 | [0.698, 0.853] | 0.902 | 0.896 | 0.777 | 0.648 | [0.471, 0.817] | 375 | fwe 0.61, vt 0.59 |

Harm = correct at full cache and wrong at b. AUC of -D, so above 0.5 is the direction the gate assumes.

### B6a-T4: DeepSeek-V2-Lite-Chat: selection against a random gate at each budget

| b | z-gate selection | 95% CI | tau=0.07 selection | 95% CI |
|---|---|---|---|---|
| 1/16 | +0.053 | [+0.022, +0.081] | +0.000 | [+0.000, +0.000] |
| 1/8 | +0.081 | [+0.053, +0.110] | +0.000 | [+0.000, +0.000] |
| 1/4 | +0.092 | [+0.065, +0.120] | +0.000 | [+0.000, +0.000] |
| 1/2 | +0.096 | [+0.069, +0.122] | +0.000 | [+0.000, +0.000] |

Bootstrap over inputs within task, 2000 replicates.

### B6b. GLM-4.7-Flash (MLA)

GLM-4.7-Flash: 200 inputs (fwe 50, niah_multikey_3 50, qa_1 50, vt 50), budgets 1/16, 1/8, 1/4, 1/2, inputs dropped for missing budgets 0, unreadable lines 0.
AUC of D for NIAH-MK3 (MK3 low) against the other three tasks: 0.993. Mean D by task: fwe +0.087, niah_multikey_3 +0.051, qa_1 +0.104, vt +0.074.

### B6b-T1: GLM-4.7-Flash: gate gain over plain SnapKV, averaged over the budget grid

| gate | open | MK3 closed | other tasks closed | mean kept | full | plain | gated | PAGE gain | random gain | selection | 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tau=0.07 | 0.685 | 0.960 | 0.100 | 0.475 | 0.925 | 0.477 | 0.734 | +0.256 | +0.141 | +0.115 | [+0.103, +0.127] |
| z-gate (0.064) | 0.765 | 0.920 | 0.007 | 0.414 | 0.925 | 0.477 | 0.703 | +0.225 | +0.105 | +0.120 | [+0.107, +0.132] |

Random gain = (1 - open) x (full - plain): a gate that opens at random at the same rate. Selection = PAGE gain - random gain. Bootstrap over inputs within task, 2000 replicates. The z-gate threshold is mean D + (-0.69) SD over all 200 inputs.

### B6b-T2: GLM-4.7-Flash: accuracy by task and budget, plain SnapKV

| task | N | mean D | full | 1/16 | 1/8 | 1/4 | 1/2 | z-gate, grid mean |
|---|---|---|---|---|---|---|---|---|
| fwe | 50 | +0.087 | 0.900 | 0.040 | 0.440 | 0.600 | 0.840 | 0.480 |
| niah_multikey_3 | 50 | +0.051 | 1.000 | 0.000 | 0.000 | 0.000 | 0.100 | 0.920 |
| qa_1 | 50 | +0.104 | 0.800 | 0.620 | 0.800 | 0.800 | 0.800 | 0.755 |
| vt | 50 | +0.074 | 1.000 | 0.000 | 0.600 | 1.000 | 1.000 | 0.655 |

### B6b-T3: GLM-4.7-Flash: D against per-input harm, with A1-T1's comparators

| model | b | full correct | harmed | pooled AUC(D) | 95% CI | task id (in-sample) | task id (split-half) | D (same splits) | within-task AUC | 95% CI | same-task pairs | per task (at least 5 in each class) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GLM-4.7-Flash | 1/16 | 185 | 153 | 0.905 | [0.853, 0.950] | 0.956 | 0.952 | 0.910 | 0.446 | [0.272, 0.631] | 386 | qa_1 0.49 |
| GLM-4.7-Flash | 1/4 | 185 | 66 | 0.848 | [0.800, 0.900] | 0.970 | 0.971 | 0.851 | 0.397 | [0.197, 0.612] | 464 | fwe 0.40 |

Harm = correct at full cache and wrong at b. AUC of -D, so above 0.5 is the direction the gate assumes.

### B6b-T4: GLM-4.7-Flash: selection against a random gate at each budget

| b | z-gate selection | 95% CI | tau=0.07 selection | 95% CI |
|---|---|---|---|---|
| 1/16 | +0.056 | [+0.046, +0.066] | +0.076 | [+0.062, +0.091] |
| 1/8 | +0.121 | [+0.104, +0.137] | +0.109 | [+0.087, +0.130] |
| 1/4 | +0.154 | [+0.137, +0.169] | +0.138 | [+0.119, +0.154] |
| 1/2 | +0.149 | [+0.127, +0.169] | +0.139 | [+0.119, +0.158] |

Bootstrap over inputs within task, 2000 replicates.

### B6-T0: B6 summary: D, gates, selection at b = 1/16, and accuracy kept by plain SnapKV

| model | design | N | full accuracy by task | AUC(D) for MK3 | tau open | tau MK3 closed | z open | z MK3 closed | z selection at 1/16 | 95% CI | kept 1/2 | kept 1/4 | kept 1/8 | kept 1/16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DeepSeek-V2-Lite-Chat | MLA | 200 | fwe 0.42, niah_multikey_3 0.62, qa_1 0.80, vt 0.52 | 1.000 | 0.000 | 1.000 | 0.750 | 1.000 | +0.053 | [+0.022, +0.081] | 0.669 | 0.576 | 0.500 | 0.305 |
| GLM-4.7-Flash | MLA | 200 | fwe 0.90, niah_multikey_3 1.00, qa_1 0.80, vt 1.00 | 0.993 | 0.685 | 0.960 | 0.765 | 0.920 | +0.056 | [+0.046, +0.066] | 0.741 | 0.649 | 0.497 | 0.178 |

Kept = plain SnapKV accuracy at b over full-cache accuracy, all four tasks. Eviction applies only to layers with a key-value cache, so for hybrids nominal budgets are not comparable with full-attention models. Kept fractions use T - 1 positions (last prompt token held back).
