# H1 spot-check report

- N examples: **200**
- Mean prompt tokens T: **5479**

## Per-budget accuracy
| budget | mean n_kept | accuracy |
|---:|---:|---:|
| 1 | 5479 | 0.790 |
| 0.875 | 4794 | 0.785 |
| 0.75 | 4109 | 0.790 |
| 0.625 | 3424 | 0.785 |
| 0.5 | 2740 | 0.785 |
| 0.375 | 2054 | 0.770 |
| 0.25 | 1370 | 0.680 |
| 0.1875 | 1027 | 0.540 |
| 0.125 | 685 | 0.265 |
| 0.0625 | 342 | 0.035 |

## H1 metric (interior accuracy maximum)
- Inputs where full-KV (b=1.0) is **wrong**: 42/200 = 0.210
- Of those, inputs where SOME budget < full-KV is **correct**: 3/200
- **rho_KV = 0.0150** (fraction of all inputs where less compute beats full-KV)

## Per-budget marginal: "correct at b, wrong at full-KV"
| budget | gain / N | loss / N | net |
|---:|---:|---:|---:|
| 0.875 | +0.000 | -0.005 | -0.005 |
| 0.75 | +0.005 | -0.005 | +0.000 |
| 0.625 | +0.010 | -0.015 | -0.005 |
| 0.5 | +0.010 | -0.015 | -0.005 |
| 0.375 | +0.005 | -0.025 | -0.020 |
| 0.25 | +0.010 | -0.120 | -0.110 |
| 0.1875 | +0.005 | -0.255 | -0.250 |
| 0.125 | +0.005 | -0.530 | -0.525 |
| 0.0625 | +0.000 | -0.755 | -0.755 |

## Interpretation
- rho_KV = 0.015 < 0.02 — H1 NOT supported on this slice. Investigate task choice / model / eviction policy before continuing.

