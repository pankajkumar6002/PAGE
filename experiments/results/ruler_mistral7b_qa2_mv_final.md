# Mistral-7B-Instruct-v0.3 RULER qa_2 + niah_multivalue sweeps (2026-07-02)

Run to fill the last four cells of the paper's partition table (these cells
previously had no data). ruler_sweep.py, N=100 per task, standard 10-budget
grid, seed 20260603, SnapKV scoring, n_sink=4, obs_window=32. 16K used
--two_pass with the sdpa-load fix added to ruler_sweep.py today.

Correctness recomputed per reanalyze_ruler.py conventions
(niah_multivalue = ALL golds; qa_2 = ANY gold).

| Task | Context | A_full | rho | n_rec/N |
|---|---|---:|---:|---|
| niah_multivalue | 4K  | 0.59 | **0.06** | 6/100 |
| niah_multivalue | 16K | 0.62 | **0.21** | 21/100 |
| qa_2            | 4K  | 0.50 | 0.01 | 1/100 |
| qa_2            | 16K | 0.56 | 0.00 | 0/100 |

- niah_multivalue crosses the pre-registered 0.05 threshold at both contexts
  and shows the context-amplification pattern (0.06 -> 0.21).
- qa_2 on Mistral is budget-insensitive at both contexts despite ~0.5
  headroom — the third QA-family exception (with QA_1-on-Mistral and
  QA_2-on-Qwen14B). Reported explicitly in the paper's partition section.

Raw: ruler_4k_mistral7b_qa2_mv.jsonl, ruler_16k_mistral7b_qa2_mv.jsonl.
