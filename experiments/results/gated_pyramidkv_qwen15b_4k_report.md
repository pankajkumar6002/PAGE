# Gated vs plain eviction analysis

- input: experiments/results/gated_pyramidkv_qwen15b_4k.jsonl
- tasks: ['fwe', 'niah_multikey_3', 'qa_1', 'vt']
- budgets: [1.0, 0.5, 0.25, 0.125]

## task = fwe
- N = 100; gate-open fraction = 1.000
| budget | plain acc | gated acc | delta (gated - plain) |
|---:|---:|---:|---:|
| 1 | 0.220 | 0.220 | +0.000 |
| 0.5 | 0.190 | 0.190 | +0.000 |
| 0.25 | 0.130 | 0.130 | +0.000 |
| 0.125 | 0.130 | 0.130 | +0.000 |

## task = niah_multikey_3
- N = 100; gate-open fraction = 0.000
| budget | plain acc | gated acc | delta (gated - plain) |
|---:|---:|---:|---:|
| 1 | 0.650 | 0.650 | +0.000 |
| 0.5 | 0.100 | 0.650 | +0.550 |
| 0.25 | 0.020 | 0.650 | +0.630 |
| 0.125 | 0.000 | 0.650 | +0.650 |

## task = qa_1
- N = 100; gate-open fraction = 1.000
| budget | plain acc | gated acc | delta (gated - plain) |
|---:|---:|---:|---:|
| 1 | 0.740 | 0.740 | +0.000 |
| 0.5 | 0.700 | 0.700 | +0.000 |
| 0.25 | 0.710 | 0.710 | +0.000 |
| 0.125 | 0.640 | 0.640 | +0.000 |

## task = vt
- N = 100; gate-open fraction = 1.000
| budget | plain acc | gated acc | delta (gated - plain) |
|---:|---:|---:|---:|
| 1 | 0.820 | 0.820 | +0.000 |
| 0.5 | 0.810 | 0.810 | +0.000 |
| 0.25 | 0.830 | 0.830 | +0.000 |
| 0.125 | 0.750 | 0.750 | +0.000 |

## Mixed suite (pooled across ['fwe', 'niah_multikey_3', 'qa_1', 'vt'])
| budget | plain mean | gated mean | delta | n_gate_open / N |
|---:|---:|---:|---:|---:|
| 1 | 0.608 | 0.608 | +0.000 | 300/400 |
| 0.5 | 0.450 | 0.588 | +0.138 | 300/400 |
| 0.25 | 0.422 | 0.580 | +0.157 | 300/400 |
| 0.125 | 0.380 | 0.542 | +0.162 | 300/400 |

## Headline (mean over all eviction budgets, mixed suite)
- plain mean accuracy: 0.417
- gated mean accuracy: 0.570
- **delta (gated - plain): +0.152**
- Week-4 hard checkpoint (≥+3pp): PASSED

