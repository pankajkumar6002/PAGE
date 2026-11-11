
## QuALITY on Qwen2.5-14B, letter and quote-then-answer arms (replication test)


### Table 47: Qwen2.5-14B on QuALITY: accuracy and selection against a random gate at the same open rate, SnapKV

| arm | model | dataset | N | docs | b | gate | open | kept | full | plain | PAGE | random | PAGE - plain | selection | 95% CI (by document) | harmed/survived | AUC(D) | AUC(a_early) | AUC(a_early) - AUC(D), 95% CI | parse failures |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| letter | Qwen2.5-14B | quality | 300 | 110 | 1/16 | tau=0.07 | 0.000 | 1.000 | 0.833 | 0.837 | 0.833 | 0.833 | -0.003 | +0.000 | [+0.000, +0.000] | 2/248 | 0.915 | 0.492 | [-0.706, -0.144] | 0.000 |
| letter | Qwen2.5-14B | quality | 300 | 110 | 1/16 | z-gate (-0.011) | 0.763 | 0.284 | 0.833 | 0.837 | 0.837 | 0.836 | +0.000 | +0.001 | [-0.010, +0.011] | 2/248 | 0.915 | 0.492 | [-0.706, -0.144] | 0.000 |
| letter | Qwen2.5-14B | quality | 300 | 110 | 1/8 | tau=0.07 | 0.000 | 1.000 | 0.833 | 0.833 | 0.833 | 0.833 | +0.000 | +0.000 | [+0.000, +0.000] | 3/247 | 0.789 | 0.628 | [-0.672, +0.320] | 0.000 |
| letter | Qwen2.5-14B | quality | 300 | 110 | 1/8 | z-gate (-0.011) | 0.763 | 0.332 | 0.833 | 0.833 | 0.830 | 0.833 | -0.003 | -0.003 | [-0.013, +0.006] | 3/247 | 0.789 | 0.628 | [-0.672, +0.320] | 0.000 |
| quote | Qwen2.5-14B | quality | 300 | 110 | 1/16 | tau=0.07 | 0.000 | 1.000 | 0.827 | 0.797 | 0.827 | 0.827 | +0.030 | +0.000 | [+0.000, +0.000] | 21/227 | 0.557 | 0.498 | [-0.212, +0.094] | 0.030 |
| quote | Qwen2.5-14B | quality | 300 | 110 | 1/16 | z-gate (-0.028) | 0.757 | 0.291 | 0.827 | 0.797 | 0.807 | 0.804 | +0.010 | +0.003 | [-0.014, +0.021] | 21/227 | 0.557 | 0.498 | [-0.212, +0.094] | 0.030 |
| quote | Qwen2.5-14B | quality | 300 | 110 | 1/8 | tau=0.07 | 0.000 | 1.000 | 0.827 | 0.823 | 0.827 | 0.827 | +0.003 | +0.000 | [+0.000, +0.000] | 12/236 | 0.496 | 0.589 | [-0.022, +0.226] | 0.010 |
| quote | Qwen2.5-14B | quality | 300 | 110 | 1/8 | z-gate (-0.028) | 0.757 | 0.338 | 0.827 | 0.823 | 0.823 | 0.824 | +0.000 | -0.001 | [-0.017, +0.014] | 12/236 | 0.496 | 0.589 | [-0.022, +0.226] | 0.010 |

As in the Beyond RULER section: bootstraps resample documents, 2000 replicates; the z-gate threshold is mean D + (-0.69) SD over the arm's inputs.

### Qwen2.5-14B on QuALITY: harm at b = 1/16, by arm

| arm | model | dataset | N | full correct | harmed | harm rate | parse failures at 1/16 | parse failures at full |
|---|---|---|---|---|---|---|---|---|
| letter | Qwen2.5-14B | quality | 300 | 250 | 2 | 0.008 | 0.000 | 0.000 |
| quote | Qwen2.5-14B | quality | 300 | 248 | 21 | 0.085 | 0.030 | 0.003 |

### Pre-registered predictions for Qwen2.5-14B on QuALITY

| prediction | test | value | verdict |
|---|---|---|---|
| P1 | quote arm, z-gate, b = 1/8: selection positive with 95% CI above 0 (the Mistral result replicates) | CI [-0.017, +0.014] | not replicated, so P1 holds |
| P2 | harm at 1/16 higher in the quote arm than in the letter arm | 0.085 vs 0.008 | holds |
| P3a | quote arm: selection CI includes 0 in the other gate and budget cells | 3/3 | holds |
| P3b | quote arm: AUC(D) against harm below 0.65, at 1/16 and 1/8 | 0.557, 0.496 | holds |
| P4 | fixed tau opens on at most 12% of inputs, both arms | letter 0.000, quote 0.000 | holds |
