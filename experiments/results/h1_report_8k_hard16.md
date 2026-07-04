# H1 spot-check report

- N examples: **200**
- Mean prompt tokens T: **5561**

## Per-budget accuracy
| budget | mean n_kept | accuracy |
|---:|---:|---:|
| 1 | 5561 | 0.695 |
| 0.875 | 4866 | 0.705 |
| 0.75 | 4171 | 0.700 |
| 0.625 | 3475 | 0.710 |
| 0.5 | 2780 | 0.665 |
| 0.375 | 2085 | 0.570 |
| 0.25 | 1390 | 0.255 |
| 0.1875 | 1042 | 0.150 |
| 0.125 | 695 | 0.050 |
| 0.0625 | 347 | 0.000 |

## H1 metric (interior accuracy maximum)
- Inputs where full-KV (b=1.0) is **wrong**: 61/200 = 0.305
- Of those, inputs where SOME budget < full-KV is **correct**: 8/200
- **rho_KV = 0.0400** (fraction of all inputs where less compute beats full-KV)

## Per-budget marginal: "correct at b, wrong at full-KV"
| budget | gain / N | loss / N | net |
|---:|---:|---:|---:|
| 0.875 | +0.010 | -0.000 | +0.010 |
| 0.75 | +0.005 | -0.000 | +0.005 |
| 0.625 | +0.015 | -0.000 | +0.015 |
| 0.5 | +0.015 | -0.045 | -0.030 |
| 0.375 | +0.025 | -0.150 | -0.125 |
| 0.25 | +0.005 | -0.445 | -0.440 |
| 0.1875 | +0.005 | -0.550 | -0.545 |
| 0.125 | +0.005 | -0.650 | -0.645 |
| 0.0625 | +0.000 | -0.695 | -0.695 |

## Interpretation
- rho_KV = 0.040 (between 0.02 and 0.05) — weak signal. Needs more examples or a different task before deciding.

