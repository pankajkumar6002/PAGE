# Gated vs plain eviction analysis

- input: experiments/results/gated_snapkv_qwen15b_4k.jsonl
- tasks: ['fwe', 'niah_multikey_3', 'qa_1', 'vt']
- budgets: [1.0, 0.5, 0.25, 0.125]

## task = fwe
- N = 100; gate-open fraction = 1.000
| budget | plain acc | gated acc | delta (gated - plain) |
|---:|---:|---:|---:|
| 1 | 0.220 | 0.220 | +0.000 |
| 0.5 | 0.210 | 0.210 | +0.000 |
| 0.25 | 0.180 | 0.180 | +0.000 |
| 0.125 | 0.120 | 0.120 | +0.000 |

## task = niah_multikey_3
- N = 100; gate-open fraction = 0.000
| budget | plain acc | gated acc | delta (gated - plain) |
|---:|---:|---:|---:|
| 1 | 0.650 | 0.650 | +0.000 |
| 0.5 | 0.140 | 0.650 | +0.510 |
| 0.25 | 0.020 | 0.650 | +0.630 |
| 0.125 | 0.000 | 0.650 | +0.650 |

## task = qa_1
- N = 100; gate-open fraction = 1.000
| budget | plain acc | gated acc | delta (gated - plain) |
|---:|---:|---:|---:|
| 1 | 0.740 | 0.740 | +0.000 |
| 0.5 | 0.730 | 0.730 | +0.000 |
| 0.25 | 0.710 | 0.710 | +0.000 |
| 0.125 | 0.670 | 0.670 | +0.000 |

## task = vt
- N = 100; gate-open fraction = 1.000
| budget | plain acc | gated acc | delta (gated - plain) |
|---:|---:|---:|---:|
| 1 | 0.820 | 0.820 | +0.000 |
| 0.5 | 0.800 | 0.800 | +0.000 |
| 0.25 | 0.820 | 0.820 | +0.000 |
| 0.125 | 0.750 | 0.750 | +0.000 |

## Mixed suite (pooled across ['fwe', 'niah_multikey_3', 'qa_1', 'vt'])
| budget | plain mean | gated mean | delta | n_gate_open / N |
|---:|---:|---:|---:|---:|
| 1 | 0.608 | 0.608 | +0.000 | 300/400 |
| 0.5 | 0.470 | 0.598 | +0.128 | 300/400 |
| 0.25 | 0.432 | 0.590 | +0.157 | 300/400 |
| 0.125 | 0.385 | 0.547 | +0.162 | 300/400 |

## Headline (mean over all eviction budgets, mixed suite)
- plain mean accuracy: 0.429
- gated mean accuracy: 0.578
- **delta (gated - plain): +0.149**
- Week-4 hard checkpoint (≥+3pp): PASSED

