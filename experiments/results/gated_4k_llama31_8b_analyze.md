# Gated vs plain eviction analysis

- input: experiments/results/gated_4k_llama31_8b.jsonl
- tasks: ['fwe', 'niah_multikey_3', 'niah_multivalue', 'qa_1', 'qa_2', 'vt']
- budgets: [1.0, 0.5, 0.25, 0.125, 0.0625]

## task = fwe
- N = 50; gate-open fraction = 1.000
| budget | plain acc | gated acc | delta (gated - plain) |
|---:|---:|---:|---:|
| 1 | 0.840 | 0.840 | +0.000 |
| 0.5 | 0.740 | 0.740 | +0.000 |
| 0.25 | 0.480 | 0.480 | +0.000 |
| 0.125 | 0.420 | 0.420 | +0.000 |
| 0.0625 | 0.040 | 0.040 | +0.000 |

## task = niah_multikey_3
- N = 50; gate-open fraction = 0.560
| budget | plain acc | gated acc | delta (gated - plain) |
|---:|---:|---:|---:|
| 1 | 1.000 | 1.000 | +0.000 |
| 0.5 | 0.100 | 0.520 | +0.420 |
| 0.25 | 0.000 | 0.440 | +0.440 |
| 0.125 | 0.000 | 0.440 | +0.440 |
| 0.0625 | 0.000 | 0.440 | +0.440 |

## task = niah_multivalue
- N = 50; gate-open fraction = 0.120
| budget | plain acc | gated acc | delta (gated - plain) |
|---:|---:|---:|---:|
| 1 | 0.940 | 0.940 | +0.000 |
| 0.5 | 0.920 | 0.940 | +0.020 |
| 0.25 | 0.920 | 0.940 | +0.020 |
| 0.125 | 0.940 | 0.940 | +0.000 |
| 0.0625 | 0.520 | 0.880 | +0.360 |

## task = qa_1
- N = 50; gate-open fraction = 0.960
| budget | plain acc | gated acc | delta (gated - plain) |
|---:|---:|---:|---:|
| 1 | 0.900 | 0.900 | +0.000 |
| 0.5 | 0.900 | 0.900 | +0.000 |
| 0.25 | 0.900 | 0.900 | +0.000 |
| 0.125 | 0.900 | 0.900 | +0.000 |
| 0.0625 | 0.860 | 0.860 | +0.000 |

## task = qa_2
- N = 50; gate-open fraction = 0.980
| budget | plain acc | gated acc | delta (gated - plain) |
|---:|---:|---:|---:|
| 1 | 0.720 | 0.720 | +0.000 |
| 0.5 | 0.740 | 0.740 | +0.000 |
| 0.25 | 0.720 | 0.720 | +0.000 |
| 0.125 | 0.720 | 0.720 | +0.000 |
| 0.0625 | 0.680 | 0.680 | +0.000 |

## task = vt
- N = 50; gate-open fraction = 1.000
| budget | plain acc | gated acc | delta (gated - plain) |
|---:|---:|---:|---:|
| 1 | 1.000 | 1.000 | +0.000 |
| 0.5 | 1.000 | 1.000 | +0.000 |
| 0.25 | 1.000 | 1.000 | +0.000 |
| 0.125 | 0.800 | 0.800 | +0.000 |
| 0.0625 | 0.000 | 0.000 | +0.000 |

## Mixed suite (pooled across ['fwe', 'niah_multikey_3', 'niah_multivalue', 'qa_1', 'qa_2', 'vt'])
| budget | plain mean | gated mean | delta | n_gate_open / N |
|---:|---:|---:|---:|---:|
| 1 | 0.900 | 0.900 | +0.000 | 231/300 |
| 0.5 | 0.733 | 0.807 | +0.073 | 231/300 |
| 0.25 | 0.670 | 0.747 | +0.077 | 231/300 |
| 0.125 | 0.630 | 0.703 | +0.073 | 231/300 |
| 0.0625 | 0.350 | 0.483 | +0.133 | 231/300 |

## Headline (mean over all eviction budgets, mixed suite)
- plain mean accuracy: 0.596
- gated mean accuracy: 0.685
- **delta (gated - plain): +0.089**
- Week-4 hard checkpoint (≥+3pp): PASSED

