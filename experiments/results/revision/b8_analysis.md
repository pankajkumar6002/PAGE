Post-hoc reporting correction (2026-11-11): two-sided bootstrap p = min(1, 2 * (min(n_nonpositive, n_nonnegative) + 1) / (R + 1)), followed by Holm adjustment. The preregistered script is unchanged. Point estimates and confidence intervals are unchanged.

## B8. PAGE on realistic near-tie retrieval mixed into L-Eval (S-Q2, G-W3, G-Q1, R-Q1)

### B8-T1: Near-tie datasets alone: accuracy by budget, harm at 1/16, and mean D against the six L-Eval tasks

| model | task | N | full | 1/16 | 1/8 | 1/4 | 1/2 | harmed/full correct | harm rate | mean D | lowest L-Eval mean D | rank of mean D (1 = lowest) | AUC(D) vs harm, within task |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-3B | nq30 | 350 | 0.546 | 0.397 | 0.451 | 0.517 | 0.540 | 76/191 | 0.398 | +0.0505 | -0.0160 | 6 of 7 | 0.481 |
| Qwen2.5-3B | babilong_qa2 | 200 | 0.310 | 0.120 | 0.200 | 0.265 | 0.295 | 44/62 | 0.710 | +0.0643 | -0.0160 | 7 of 7 | 0.480 |
| Qwen2.5-3B | babilong_qa3 | 200 | 0.285 | 0.180 | 0.265 | 0.265 | 0.265 | 36/57 | 0.632 | +0.0667 | -0.0160 | 7 of 7 | 0.574 |
| Mistral-7B | nq30 | 350 | 0.591 | 0.517 | 0.577 | 0.566 | 0.589 | 53/207 | 0.256 | +0.0735 | +0.0415 | 6 of 7 | 0.480 |
| Mistral-7B | babilong_qa2 | 200 | 0.115 | 0.045 | 0.110 | 0.125 | 0.100 | 18/23 | 0.783 | +0.0897 | +0.0415 | 6 of 7 | 0.211 |
| Mistral-7B | babilong_qa3 | 200 | 0.240 | 0.185 | 0.240 | 0.250 | 0.245 | 24/48 | 0.500 | +0.0815 | +0.0415 | 6 of 7 | 0.656 |
| Qwen2.5-14B | nq30 | 350 | 0.649 | 0.597 | 0.631 | 0.646 | 0.654 | 36/227 | 0.159 | +0.0804 | -0.0029 | 7 of 7 | 0.413 |
| Qwen2.5-14B | babilong_qa2 | 200 | 0.510 | 0.445 | 0.540 | 0.520 | 0.515 | 18/102 | 0.176 | +0.0794 | -0.0029 | 7 of 7 | 0.389 |
| Qwen2.5-14B | babilong_qa3 | 200 | 0.400 | 0.380 | 0.375 | 0.410 | 0.405 | 14/80 | 0.175 | +0.0855 | -0.0029 | 7 of 7 | 0.473 |

Harm = correct at full cache and wrong under plain SnapKV at 1/16. AUC reads low D as more harm. Gate-free screen (pre-registered): a dataset is capacity-bound for a model if its pooled harm rate is at least 0.15.

### B8-T2: Mixed workloads (near-tie dataset + the model's 920 L-Eval questions): selection against a random gate

| model | near-tie dataset | near-tie share | b | gate | open | closed on near-tie | closed on L-Eval | full | plain | PAGE | selection | 95% CI | bootstrap p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-3B | nq30 | 0.276 | 1/16 | z | 0.762 | 0.026 | 0.318 | 0.566 | 0.526 | 0.520 | -0.016 | [-0.023, -0.009] | 0.0010 |
| Qwen2.5-3B | nq30 | 0.276 | 1/16 | tau | 0.050 | 0.854 | 0.987 | 0.566 | 0.526 | 0.564 | -0.000 | [-0.005, +0.004] | 0.8966 |
| Qwen2.5-3B | nq30 | 0.276 | 1/8 | z | 0.762 | 0.026 | 0.318 | 0.566 | 0.550 | 0.544 | -0.009 | [-0.017, -0.002] | 0.0170 |
| Qwen2.5-3B | nq30 | 0.276 | 1/8 | tau | 0.050 | 0.854 | 0.987 | 0.566 | 0.550 | 0.565 | -0.001 | [-0.006, +0.004] | 0.7736 |
| Qwen2.5-3B | babilong | 0.303 | 1/16 | z | 0.742 | 0.005 | 0.367 | 0.490 | 0.446 | 0.439 | -0.019 | [-0.026, -0.012] | 0.0010 |
| Qwen2.5-3B | babilong | 0.303 | 1/16 | tau | 0.129 | 0.605 | 0.987 | 0.490 | 0.446 | 0.472 | -0.013 | [-0.021, -0.005] | 0.0010 |
| Qwen2.5-3B | babilong | 0.303 | 1/8 | z | 0.742 | 0.005 | 0.367 | 0.490 | 0.480 | 0.474 | -0.008 | [-0.015, -0.001] | 0.0300 |
| Qwen2.5-3B | babilong | 0.303 | 1/8 | tau | 0.129 | 0.605 | 0.987 | 0.490 | 0.480 | 0.486 | -0.003 | [-0.010, +0.003] | 0.3448 |
| Mistral-7B | nq30 | 0.276 | 1/16 | z | 0.776 | 0.023 | 0.300 | 0.598 | 0.537 | 0.544 | -0.006 | [-0.013, +0.002] | 0.1299 |
| Mistral-7B | nq30 | 0.276 | 1/16 | tau | 0.309 | 0.429 | 0.790 | 0.598 | 0.537 | 0.564 | -0.015 | [-0.025, -0.006] | 0.0030 |
| Mistral-7B | nq30 | 0.276 | 1/8 | z | 0.776 | 0.023 | 0.300 | 0.598 | 0.576 | 0.579 | -0.002 | [-0.008, +0.006] | 0.7586 |
| Mistral-7B | nq30 | 0.276 | 1/8 | tau | 0.309 | 0.429 | 0.790 | 0.598 | 0.576 | 0.588 | -0.003 | [-0.011, +0.006] | 0.5267 |
| Mistral-7B | babilong | 0.303 | 1/16 | z | 0.742 | 0.005 | 0.368 | 0.472 | 0.414 | 0.426 | -0.004 | [-0.010, +0.003] | 0.3058 |
| Mistral-7B | babilong | 0.303 | 1/16 | tau | 0.394 | 0.182 | 0.790 | 0.472 | 0.414 | 0.436 | -0.013 | [-0.022, -0.004] | 0.0060 |
| Mistral-7B | babilong | 0.303 | 1/8 | z | 0.742 | 0.005 | 0.368 | 0.472 | 0.454 | 0.461 | +0.003 | [-0.004, +0.009] | 0.4238 |
| Mistral-7B | babilong | 0.303 | 1/8 | tau | 0.394 | 0.182 | 0.790 | 0.472 | 0.454 | 0.468 | +0.003 | [-0.003, +0.010] | 0.3548 |
| Qwen2.5-14B | nq30 | 0.276 | 1/16 | z | 0.728 | 0.000 | 0.376 | 0.765 | 0.721 | 0.718 | -0.015 | [-0.019, -0.011] | 0.0010 |
| Qwen2.5-14B | nq30 | 0.276 | 1/16 | tau | 0.346 | 0.297 | 0.790 | 0.765 | 0.721 | 0.730 | -0.020 | [-0.029, -0.012] | 0.0010 |
| Qwen2.5-14B | nq30 | 0.276 | 1/8 | z | 0.728 | 0.000 | 0.376 | 0.765 | 0.754 | 0.755 | -0.002 | [-0.006, +0.002] | 0.2449 |
| Qwen2.5-14B | nq30 | 0.276 | 1/8 | tau | 0.346 | 0.297 | 0.790 | 0.765 | 0.754 | 0.757 | -0.004 | [-0.011, +0.002] | 0.2369 |
| Qwen2.5-14B | babilong | 0.303 | 1/16 | z | 0.725 | 0.005 | 0.392 | 0.702 | 0.661 | 0.658 | -0.014 | [-0.018, -0.010] | 0.0010 |
| Qwen2.5-14B | babilong | 0.303 | 1/16 | tau | 0.376 | 0.242 | 0.790 | 0.702 | 0.661 | 0.670 | -0.016 | [-0.024, -0.009] | 0.0010 |
| Qwen2.5-14B | babilong | 0.303 | 1/8 | z | 0.725 | 0.005 | 0.392 | 0.702 | 0.697 | 0.698 | -0.001 | [-0.004, +0.003] | 0.7836 |
| Qwen2.5-14B | babilong | 0.303 | 1/8 | tau | 0.376 | 0.242 | 0.790 | 0.702 | 0.697 | 0.698 | -0.003 | [-0.008, +0.002] | 0.3188 |

z threshold = mean D + (-0.69) SD over the whole mixed workload, re-fit in every bootstrap replicate. Bootstrap: within each task, documents resampled (NQ and BABILong: each question; gsm100: each question), 2000 replicates.

### B8-T3: Primary test, z-gate at 1/16, Holm across the (model, dataset) tests

| model | near-tie dataset | selection | CI lower | Holm-adjusted p | harm screen passed |
|---|---|---|---|---|---|
| Qwen2.5-3B | nq30 | -0.016 | -0.023 | 0.0060 | yes |
| Qwen2.5-3B | babilong | -0.019 | -0.026 | 0.0060 | yes |
| Mistral-7B | nq30 | -0.006 | -0.013 | 0.2599 | yes |
| Mistral-7B | babilong | -0.004 | -0.010 | 0.3058 | yes |
| Qwen2.5-14B | nq30 | -0.015 | -0.019 | 0.0060 | yes |
| Qwen2.5-14B | babilong | -0.014 | -0.018 | 0.0060 | yes |

- nq30: positive selection with Holm-adjusted p < 0.05 on 0 of 3 models. Pre-registered win: at least 2 of 3. no win.

- babilong: positive selection with Holm-adjusted p < 0.05 on 0 of 3 models. Pre-registered win: at least 2 of 3. no win.

### B8-T4: Near-tie share sweep, z-gate, 1/16 (mean selection over subsample draws)

| model | near-tie dataset | near-tie share | mean selection | 2.5 to 97.5 percentile over draws |
|---|---|---|---|---|
| Qwen2.5-3B | nq30 | 0.10 | -0.010 | [-0.012, -0.008] |
| Qwen2.5-3B | nq30 | 0.25 | -0.016 | [-0.017, -0.014] |
| Qwen2.5-3B | nq30 | 0.50 | -0.021 | [-0.028, -0.014] |
| Qwen2.5-3B | babilong | 0.10 | -0.011 | [-0.013, -0.009] |
| Qwen2.5-3B | babilong | 0.25 | -0.015 | [-0.017, -0.014] |
| Qwen2.5-3B | babilong | 0.50 | -0.020 | [-0.026, -0.016] |
| Mistral-7B | nq30 | 0.10 | -0.004 | [-0.006, -0.003] |
| Mistral-7B | nq30 | 0.25 | -0.007 | [-0.008, -0.006] |
| Mistral-7B | nq30 | 0.50 | -0.010 | [-0.016, -0.005] |
| Mistral-7B | babilong | 0.10 | -0.005 | [-0.006, -0.003] |
| Mistral-7B | babilong | 0.25 | -0.004 | [-0.005, -0.003] |
| Mistral-7B | babilong | 0.50 | -0.005 | [-0.011, +0.000] |
| Qwen2.5-14B | nq30 | 0.10 | -0.015 | [-0.017, -0.014] |
| Qwen2.5-14B | nq30 | 0.25 | -0.015 | [-0.016, -0.014] |
| Qwen2.5-14B | nq30 | 0.50 | -0.013 | [-0.016, -0.011] |
| Qwen2.5-14B | babilong | 0.10 | -0.015 | [-0.017, -0.014] |
| Qwen2.5-14B | babilong | 0.25 | -0.015 | [-0.016, -0.013] |
| Qwen2.5-14B | babilong | 0.50 | -0.011 | [-0.014, -0.009] |

The interval here is the spread over subsample draws, not a sampling CI.

### B8-T5: Near-tie dataset alone, z threshold within the dataset (per-input selection)

| model | dataset | b | open | selection | 95% CI |
|---|---|---|---|---|---|
| Qwen2.5-3B | nq30 | 1/16 | 0.763 | -0.001 | [-0.026, +0.022] |
| Qwen2.5-3B | nq30 | 1/8 | 0.763 | +0.000 | [-0.020, +0.021] |
| Qwen2.5-3B | babilong | 1/16 | 0.767 | +0.001 | [-0.022, +0.020] |
| Qwen2.5-3B | babilong | 1/8 | 0.767 | +0.002 | [-0.012, +0.018] |
| Mistral-7B | nq30 | 1/16 | 0.766 | -0.012 | [-0.032, +0.012] |
| Mistral-7B | nq30 | 1/8 | 0.766 | -0.000 | [-0.017, +0.021] |
| Mistral-7B | babilong | 1/16 | 0.767 | +0.008 | [-0.009, +0.023] |
| Mistral-7B | babilong | 1/8 | 0.767 | +0.009 | [-0.004, +0.023] |
| Qwen2.5-14B | nq30 | 1/16 | 0.769 | -0.009 | [-0.029, +0.007] |
| Qwen2.5-14B | nq30 | 1/8 | 0.769 | -0.007 | [-0.020, +0.007] |
| Qwen2.5-14B | babilong | 1/16 | 0.767 | +0.000 | [-0.014, +0.012] |
| Qwen2.5-14B | babilong | 1/8 | 0.767 | -0.007 | [-0.016, +0.001] |

### B8-T6: Pre-registered predictions (scored by hand from B8-T1 to B8-T5, 2026-11-10)

| ID | Prediction | Result | Verdict |
|---|---|---|---|
| P1 | nq30 passes the harm screen (>= 0.15) on >= 2 of 3 models | 0.398, 0.256, 0.159: 3 of 3 | holds |
| P2 | babilong passes the harm screen on >= 2 of 3 models | qa2/qa3 pooled passes on 3 of 3 (task rates 0.175 to 0.783) | holds |
| P3 | nq30 has the lowest mean D of the 7 tasks on >= 2 of 3 models | rank 6, 6, 7 of 7: it has the highest or second-highest D | fails, in the opposite direction |
| P4 | primary test is a win on nq30 (stated at ~30%; no win expected) | 0 of 3 models positive; Holm-significant negative on 2 | no win, as expected |
| P5 | dataset alone, z within: CI includes 0 on most units at 1/16 | 6 of 6 | holds |

Pre-registered outcome: **no win on either dataset.** Both near-tie datasets are capacity-bound by the gate-free screen (SnapKV at 1/16 loses 16% to 78% of full-correct answers), but their mean D is the highest in the workload on every model, so the z-gate closes on 0% to 2.6% of them and on 30% to 39% of L-Eval questions, which eviction barely harms. Selection is negative on all six primary units and does not turn positive at any near-tie share (B8-T4). Within each dataset D is at chance against harm (AUC 0.21 to 0.66, B8-T1; B8-T5).
