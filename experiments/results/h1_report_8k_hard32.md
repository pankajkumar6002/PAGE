# H1 spot-check report

- N examples: **200**
- Mean prompt tokens T: **5726**

## Per-budget accuracy
| budget | mean n_kept | accuracy |
|---:|---:|---:|
| 1 | 5726 | 0.760 |
| 0.875 | 5010 | 0.760 |
| 0.75 | 4294 | 0.745 |
| 0.625 | 3579 | 0.660 |
| 0.5 | 2863 | 0.450 |
| 0.375 | 2147 | 0.195 |
| 0.25 | 1431 | 0.060 |
| 0.1875 | 1073 | 0.025 |
| 0.125 | 715 | 0.015 |
| 0.0625 | 357 | 0.005 |

## H1 metric (interior accuracy maximum)
- Inputs where full-KV (b=1.0) is **wrong**: 48/200 = 0.240
- Of those, inputs where SOME budget < full-KV is **correct**: 2/200
- **rho_KV = 0.0100** (fraction of all inputs where less compute beats full-KV)

## Per-budget marginal: "correct at b, wrong at full-KV"
| budget | gain / N | loss / N | net |
|---:|---:|---:|---:|
| 0.875 | +0.005 | -0.005 | +0.000 |
| 0.75 | +0.010 | -0.025 | -0.015 |
| 0.625 | +0.005 | -0.105 | -0.100 |
| 0.5 | +0.005 | -0.315 | -0.310 |
| 0.375 | +0.005 | -0.570 | -0.565 |
| 0.25 | +0.005 | -0.705 | -0.700 |
| 0.1875 | +0.000 | -0.735 | -0.735 |
| 0.125 | +0.000 | -0.745 | -0.745 |
| 0.0625 | +0.000 | -0.755 | -0.755 |

## Interpretation
- rho_KV = 0.010 < 0.02 — H1 NOT supported on this slice. Investigate task choice / model / eviction policy before continuing.

