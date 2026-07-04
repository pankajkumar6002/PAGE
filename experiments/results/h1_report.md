# H1 spot-check report

- N examples: **150**
- Mean prompt tokens T: **2739**

## Per-budget accuracy
| budget | mean n_kept | accuracy |
|---:|---:|---:|
| 1 | 2739 | 0.827 |
| 0.875 | 2396 | 0.820 |
| 0.75 | 2054 | 0.820 |
| 0.625 | 1712 | 0.827 |
| 0.5 | 1369 | 0.833 |
| 0.375 | 1027 | 0.807 |
| 0.25 | 684 | 0.760 |
| 0.1875 | 513 | 0.547 |
| 0.125 | 342 | 0.207 |
| 0.0625 | 171 | 0.033 |

## H1 metric (interior accuracy maximum)
- Inputs where full-KV (b=1.0) is **wrong**: 26/150 = 0.173
- Of those, inputs where SOME budget < full-KV is **correct**: 2/150
- **rho_KV = 0.0133** (fraction of all inputs where less compute beats full-KV)

## Per-budget marginal: "correct at b, wrong at full-KV"
| budget | gain / N | loss / N | net |
|---:|---:|---:|---:|
| 0.875 | +0.000 | -0.007 | -0.007 |
| 0.75 | +0.000 | -0.007 | -0.007 |
| 0.625 | +0.000 | -0.000 | +0.000 |
| 0.5 | +0.007 | -0.000 | +0.007 |
| 0.375 | +0.007 | -0.027 | -0.020 |
| 0.25 | +0.007 | -0.073 | -0.067 |
| 0.1875 | +0.007 | -0.287 | -0.280 |
| 0.125 | +0.000 | -0.620 | -0.620 |
| 0.0625 | +0.000 | -0.793 | -0.793 |

## Interpretation
- rho_KV = 0.013 < 0.02 — H1 NOT supported on this slice. Investigate task choice / model / eviction policy before continuing.

