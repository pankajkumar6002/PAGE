# Partition table cell: QA_2 @ Qwen2.5-3B-Instruct, 16K context

Confirmatory run to fill the one `--` cell in `tab:partition` (QA_2, Qwen2.5-3B, 16384).

## Setup
- Model: `Qwen/Qwen2.5-3B-Instruct`
- RULER config: 16384 (16K), task `qa_2`
- Examples: N = 100 (`--max_examples 100`, no reduction needed; fit fine)
- Scoring: SnapKV two-pass (`--two_pass`), sdpa attention, GPU 1, bf16
- Budgets swept: 1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.1875, 0.125, 0.0625
- Runtime: 312s, exit 0, no OOM/crash. All 100 examples completed.
- Raw: `experiments/results/ruler_16k_qwen3b_qa2.jsonl` (1000 records)
- Log: `experiments/logs/ruler_16k_qwen3b_qa2.log`

## Analysis (canonical convention)
`is_correct`: qa_2 = ANY gold substring in pred, lowercased
(`experiments/scripts/reanalyze_ruler.py`).

rho = fraction of inputs where full-KV (b=1.0) is WRONG and some budget b<1.0 is CORRECT.

| quantity | value |
|---|---|
| **rho_KV** | **0.0200** (2/100) |
| **A_full** (b=1.0 accuracy) | **0.410** |
| N | 100 |
| wrong@full | 59 |

## Threshold
rho = 0.0200 is **below** the 0.05 dilution-prone threshold. This cell is NOT dilution-prone.

## Cross-check against neighboring cells
- QA_2 @ Qwen3B-4K: rho = 0.04 (paper). Our 16K value (0.02) is the same order of magnitude, slightly lower.
- QA_1 @ Qwen3B-16K: rho = 0.08 (paper). Our QA_2 16K value (0.02) is lower, consistent with QA_2 being the harder/lower-signal task.

Both neighbors are consistent with rho_KV = 0.02 here.

## Table cell
**rho = 0.02**  (A_full = 0.41, N = 100)
