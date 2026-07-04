# H1 spot-check report

- N examples: **27**
- Mean prompt tokens T: **4010**

## Per-budget accuracy
| budget | mean n_kept | accuracy |
|---:|---:|---:|
| 1 | 4010 | 0.704 |
| 0.875 | 3508 | 0.556 |
| 0.75 | 3007 | 0.444 |
| 0.625 | 2413 | 0.296 |
| 0.5 | 1931 | 0.074 |
| 0.375 | 1448 | 0.074 |
| 0.25 | 965 | 0.037 |
| 0.1875 | 724 | 0.000 |
| 0.125 | 482 | 0.000 |
| 0.0625 | 241 | 0.000 |

## H1 metric (interior accuracy maximum)
- Inputs where full-KV (b=1.0) is **wrong**: 8/27 = 0.296
- Of those, inputs where SOME budget < full-KV is **correct**: 0/27
- **rho_KV = 0.0000** (fraction of all inputs where less compute beats full-KV)

## Per-budget marginal: "correct at b, wrong at full-KV"
| budget | gain / N | loss / N | net |
|---:|---:|---:|---:|
| 0.875 | +0.000 | -0.148 | -0.148 |
| 0.75 | +0.000 | -0.259 | -0.259 |
| 0.625 | +0.000 | -0.407 | -0.407 |
| 0.5 | +0.000 | -0.630 | -0.630 |
| 0.375 | +0.000 | -0.630 | -0.630 |
| 0.25 | +0.000 | -0.667 | -0.667 |
| 0.1875 | +0.000 | -0.704 | -0.704 |
| 0.125 | +0.000 | -0.704 | -0.704 |
| 0.0625 | +0.000 | -0.704 | -0.704 |

## Interpretation
- rho_KV = 0.000 < 0.02 — H1 NOT supported on this slice. Investigate task choice / model / eviction policy before continuing.

