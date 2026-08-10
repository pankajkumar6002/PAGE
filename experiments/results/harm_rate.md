# Fixed-budget harm rate, tau = 0.07

Pr_x[ A(b, x) < A_full(x) ]: the fraction of inputs an arm gets wrong that the full cache gets right.

## Corrected pool (7 SnapKV cells) - PROMOTED TO THE PAPER

| task | b=0.0625 gated / plain | b=0.125 gated / plain | b=0.25 gated / plain | b=0.375 gated / plain | b=0.5 gated / plain | b=0.625 gated / plain | b=0.75 gated / plain | b=0.875 gated / plain |
|---|---|---|---|---|---|---|---|---|
| NIAH-MK3 | 0.026 / 0.750 | 0.026 / 0.744 | 0.024 / 0.724 | 0.022 / 0.658 | 0.018 / 0.546 | 0.018 / 0.448 | 0.018 / 0.314 | 0.010 / 0.166 |
| FWE | 0.236 / 0.414 | 0.162 / 0.298 | 0.118 / 0.224 | 0.086 / 0.156 | 0.064 / 0.116 | 0.044 / 0.076 | 0.026 / 0.044 | 0.018 / 0.026 |
| QA_1 | 0.110 / 0.110 | 0.064 / 0.064 | 0.030 / 0.030 | 0.018 / 0.018 | 0.010 / 0.010 | 0.016 / 0.016 | 0.014 / 0.014 | 0.014 / 0.014 |
| VT | 0.684 / 0.694 | 0.108 / 0.110 | 0.034 / 0.036 | 0.026 / 0.030 | 0.026 / 0.026 | 0.028 / 0.030 | 0.018 / 0.022 | 0.012 / 0.014 |

N per (task, budget): min 500, max 500

Cells pooled:
- `gated_4k_qwen15b.jsonl` (3600 rows)
- `gated_4k_qwen3b.jsonl` (3600 rows)
- `gated_4k_qwen14b.jsonl` (1800 rows)
- `gated_4k_mistral7b.jsonl` (3600 rows)
- `gated_16k_qwen3b.jsonl` (1800 rows)
- `gated_16k_mistral7b.jsonl` (1800 rows)
- `gated_16k_qwen15b_sdpa.jsonl` (1800 rows)

NIAH-MK3 at b = 0.0625: harm falls 0.750 -> 0.026, a 29x reduction. Harm is reduced, not eliminated.
Gate inert at b = 0.0625 (gated == plain): QA_1
VT at b = 0.0625 is harmed either way: 0.694 plain vs 0.684 gated.

## The revision plan's pool, for provenance

Two differences from the corrected pool, both of which move the numbers:

1. It substitutes `gated_16k_qwen15b.jsonl` (9 rows, NIAH-MK3 only, one input) for the 1800-row `gated_16k_qwen15b_sdpa.jsonl`, so the 16K Qwen2.5-1.5B cell is absent from FWE, QA_1 and VT and N falls to about 450.
2. It reads `correct_gated` off the log instead of re-evaluating at tau = 0.07, so the Qwen2.5-3B 4K cell enters at its executed tau = 0.04. This lowers the gated harm rate on FWE and VT only.

| task | b=0.0625 gated / plain | b=0.5 gated / plain |
|---|---|---|
| NIAH-MK3 | 0.029 / 0.796 | 0.020 / 0.594 |
| FWE | 0.291 / 0.444 | 0.071 / 0.120 |
| QA_1 | 0.118 / 0.118 | 0.009 / 0.009 |
| VT | 0.751 / 0.756 | 0.022 / 0.022 |

N per (task, budget): min 450, max 451
