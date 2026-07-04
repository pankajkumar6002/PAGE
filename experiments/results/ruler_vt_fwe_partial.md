# H1 spot-check report

- N examples: **82**
- Mean prompt tokens T: **3876**

## Per-budget accuracy
| budget | mean n_kept | accuracy |
|---:|---:|---:|
| 1 | 3876 | 1.000 |
| 0.875 | 3350 | 0.988 |
| 0.75 | 2871 | 0.988 |
| 0.625 | 2393 | 0.988 |
| 0.5 | 1914 | 0.988 |
| 0.375 | 1435 | 0.988 |
| 0.25 | 957 | 0.988 |
| 0.1875 | 717 | 0.988 |
| 0.125 | 478 | 0.988 |
| 0.0625 | 239 | 0.988 |

## H1 metric (interior accuracy maximum)
- Inputs where full-KV (b=1.0) is **wrong**: 0/82 = 0.000
- Of those, inputs where SOME budget < full-KV is **correct**: 0/82
- **rho_KV = 0.0000** (fraction of all inputs where less compute beats full-KV)

## Per-budget marginal: "correct at b, wrong at full-KV"
| budget | gain / N | loss / N | net |
|---:|---:|---:|---:|
| 0.875 | +0.000 | -0.012 | -0.012 |
| 0.75 | +0.000 | -0.012 | -0.012 |
| 0.625 | +0.000 | -0.012 | -0.012 |
| 0.5 | +0.000 | -0.012 | -0.012 |
| 0.375 | +0.000 | -0.012 | -0.012 |
| 0.25 | +0.000 | -0.012 | -0.012 |
| 0.1875 | +0.000 | -0.012 | -0.012 |
| 0.125 | +0.000 | -0.012 | -0.012 |
| 0.0625 | +0.000 | -0.012 | -0.012 |

## Interpretation
- rho_KV = 0.000 < 0.02 — H1 NOT supported on this slice. Investigate task choice / model / eviction policy before continuing.

