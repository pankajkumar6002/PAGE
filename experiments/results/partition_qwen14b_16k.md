# Partition (per-task recovery rate) -- Qwen2.5-14B-Instruct @ RULER 16K

- Source files:
  - /home/smlab/projects/eff-nn/experiments/results/gated_16k_qwen14b.jsonl
  - /home/smlab/projects/eff-nn/experiments/results/gated_16k_qwen14b_extra.jsonl
- Metric: rho = Pr_x[some b<1.0 is correct AND b=1.0 is wrong]  (strict per-input Pareto).
- N = 30 per task.

| Task | rho (plain) | rho (gated) | full-KV acc (plain) | Predicted class | Predicted passes |
|------|-------------|-------------|---------------------|-----------------|------------------|
| VT | 0.03 | 0.00 | 0.97 | dilution-prone | no |
| FWE | 0.20 | 0.03 | 0.80 | dilution-prone | yes |
| QA\_1 | 0.10 | 0.10 | 0.67 | dilution-prone | yes |
| QA\_2 | 0.03 | 0.03 | 0.67 | dilution-prone | no |
| niah\_multivalue | 0.07 | 0.07 | 0.63 | dilution-prone | yes |
| NIAH-MK3 | 0.00 | 0.00 | 1.00 | capacity-bound | yes |

## Per-task summary (raw numbers)

- vt: N=30, rho_plain=0.0333, rho_gated=0.0000, full_acc_plain=0.9667, full_acc_gated=0.9667, class=dilution-prone
- fwe: N=30, rho_plain=0.2000, rho_gated=0.0333, full_acc_plain=0.8000, full_acc_gated=0.8000, class=dilution-prone
- qa_1: N=30, rho_plain=0.1000, rho_gated=0.1000, full_acc_plain=0.6667, full_acc_gated=0.6667, class=dilution-prone
- qa_2: N=30, rho_plain=0.0333, rho_gated=0.0333, full_acc_plain=0.6667, full_acc_gated=0.6667, class=dilution-prone
- niah_multivalue: N=30, rho_plain=0.0667, rho_gated=0.0667, full_acc_plain=0.6333, full_acc_gated=0.6333, class=dilution-prone
- niah_multikey_3: N=30, rho_plain=0.0000, rho_gated=0.0000, full_acc_plain=1.0000, full_acc_gated=1.0000, class=capacity-bound

## Paper-ready column for tab:partition
Column header: `Qwen2.5-14B 16K`. Bold marks rho >= 0.05.

```
Task              | Qwen2.5-14B 16K
------------------|----------------
VT                | 0.03
FWE               | \textbf{0.20}
QA\_1             | \textbf{0.10}
QA\_2             | 0.03
niah\_multivalue  | \textbf{0.07}
NIAH-MK3          | 0.00
```

## Partition verdict
- 5 dilution-prone tasks all have rho >= 0.05: NO
- NIAH-MK3 has rho <= 0.02: YES
- Partition PARTIALLY HOLDS: capacity-bound side clean; dilution-prone failure(s) on ['vt', 'qa_2'].

## 4K vs 16K comparison (this model)

rho values from /home/smlab/projects/eff-nn/experiments/results/partition_qwen14b_4k.md:

| Task | 4K rho (plain) | 16K rho (plain) | direction |
|------|----------------|------------------|-----------|
| VT | 0.00 | 0.03 | up |
| FWE | 0.00 | 0.20 | up |
| QA\_1 | 0.06 | 0.10 | up |
| QA\_2 | 0.02 | 0.03 | flat |
| niah\_multivalue | 0.08 | 0.07 | flat |
| NIAH-MK3 | 0.00 | 0.00 | flat |

Reading: at 4K, Qwen2.5-14B saturates A_full on VT/FWE/NIAH-MK3 and QA_2, so rho is mechanically near zero (no wrong-at-full inputs to recover). The scaling-formula prediction is that at 16K, A_full drops for the dilution-prone tasks (longer context = more distractors), headroom opens, and rho should rise on those tasks while NIAH-MK3 stays near zero (still capacity-bound). The table above is the verification: directions marked `up` confirm re-entry to the dilution regime.
