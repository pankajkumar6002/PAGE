# Gated vs plain eviction analysis

- input: experiments/results/gated_h2o_qwen15b_4k.jsonl
- tasks: ['fwe', 'niah_multikey_3', 'qa_1', 'vt']
- budgets: [1.0, 0.5, 0.25, 0.125]

## task = fwe
- N = 100; gate-open fraction = 1.000
| budget | plain acc | gated acc | delta (gated - plain) |
|---:|---:|---:|---:|
| 1 | 0.220 | 0.220 | +0.000 |
| 0.5 | 0.230 | 0.230 | +0.000 |
| 0.25 | 0.200 | 0.200 | +0.000 |
| 0.125 | 0.120 | 0.120 | +0.000 |

## task = niah_multikey_3
- N = 100; gate-open fraction = 0.000
| budget | plain acc | gated acc | delta (gated - plain) |
|---:|---:|---:|---:|
| 1 | 0.650 | 0.650 | +0.000 |
| 0.5 | 0.010 | 0.650 | +0.640 |
| 0.25 | 0.000 | 0.650 | +0.650 |
| 0.125 | 0.000 | 0.650 | +0.650 |

## task = qa_1
- N = 100; gate-open fraction = 1.000
| budget | plain acc | gated acc | delta (gated - plain) |
|---:|---:|---:|---:|
| 1 | 0.740 | 0.740 | +0.000 |
| 0.5 | 0.290 | 0.290 | +0.000 |
| 0.25 | 0.240 | 0.240 | +0.000 |
| 0.125 | 0.150 | 0.150 | +0.000 |

## task = vt
- N = 100; gate-open fraction = 1.000
| budget | plain acc | gated acc | delta (gated - plain) |
|---:|---:|---:|---:|
| 1 | 0.820 | 0.820 | +0.000 |
| 0.5 | 0.660 | 0.660 | +0.000 |
| 0.25 | 0.410 | 0.410 | +0.000 |
| 0.125 | 0.040 | 0.040 | +0.000 |

## Mixed suite (pooled across ['fwe', 'niah_multikey_3', 'qa_1', 'vt'])
| budget | plain mean | gated mean | delta | n_gate_open / N |
|---:|---:|---:|---:|---:|
| 1 | 0.608 | 0.608 | +0.000 | 300/400 |
| 0.5 | 0.297 | 0.458 | +0.160 | 300/400 |
| 0.25 | 0.212 | 0.375 | +0.163 | 300/400 |
| 0.125 | 0.077 | 0.240 | +0.162 | 300/400 |

## Headline (mean over all eviction budgets, mixed suite)
- plain mean accuracy: 0.196
- gated mean accuracy: 0.357
- **delta (gated - plain): +0.162**
- Week-4 hard checkpoint (≥+3pp): PASSED

