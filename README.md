# PAGE: Partition-Aware Gated KV-Cache Eviction

A training-free, label-free gate that decides, **per input and before decoding, whether to
evict the KV cache at all**. It computes one scalar from the prefill attention — the
early-to-late-layer drop in cross-layer attention-head agreement — and runs any SnapKV-style
evictor when the drop is large, otherwise keeps the full cache. Read honestly, PAGE is a
**safety mechanism**: it prevents catastrophic accuracy collapse on capacity-bound inputs
(precise multi-key retrieval, exact code completion) at a bounded memory cost, and improves the
accuracy–memory frontier where moderate compression is the target.

This repository accompanies the ICLR-format paper in [`paper/iclr2026/`](paper/iclr2026/)
(compiled: [`main.pdf`](paper/iclr2026/main.pdf)).

> **Status.** Research code and a submission-format draft. The method is deliberately scoped:
> it is a moderate-compression (≤ ~3×) safety wrapper, not an aggressive-regime compressor, and
> it does not beat strong *trained* evictors at matched memory. See [Limitations](#limitations).

## What it does

- **A task-type partition.** Eviction *helps* dilution-prone tasks (multi-hop tracking,
  aggregation, QA, multi-value) and *catastrophically hurts* capacity-bound ones (RULER
  NIAH-MultiKey-3; LongBench `lcc` code completion). NIAH-MK3 is capacity-bound independently
  of the scoring family (confirmed against a faithful ManifoldKV+Ada-KV reproduction).
- **An a-priori predictor.** The head-agreement drop `D` predicts which side an input falls on,
  from one prefill pass, with no labels and no training. It is the only one of six cheap
  prefill statistics that separates the partition (per-input AUC 1.000 on the calibration cell).
- **The gate (PAGE).** `if D >= tau: run base evictor; else: keep full cache`. A drop-in wrapper
  around SnapKV, H2O, StreamingLLM, PyramidKV, CapKV, or DBTrimKV.
- **A scaling relation.** `rho ~ (1 - A_full(M,T)) * p_recov(...)` for the fraction of inputs
  that recover; stress-tested out-of-range at 32K (directional/ordinal content holds,
  constant-ratio proportionality does not).

## Headline results (honest framing)

- **Safety showcase.** Mistral-7B, RULER 4K NIAH-MK3: plain SnapKV falls 99% → 0% as the budget
  shrinks; PAGE holds ~89% at every budget.
- **Realistic capacity-bound exemplar.** LongBench `lcc` code completion (Qwen2.5-14B): plain
  eviction destroys 64% of correct answers (0.29 → 0.14); the gate closes a priori and protects
  accuracy (+0.146 over plain).
- **Accuracy–memory frontier.** At matched *achieved* cache, PAGE improves on plain eviction by
  3–19 pp above a ~0.32–0.35 kept-KV crossover; below it, and versus a strong trained method at
  matched memory, PAGE does not win.
- **Transfer.** One fixed threshold works on the Qwen2.5 and Mistral families; an unlabeled
  per-model z-scoring pilot extends it to Llama-family and Qwen3 models.

## Repository layout

```
paper/
  main.tex                 full-content master (NeurIPS-stub preamble)
  iclr2026/main.tex        ICLR 2026 submission build (9-page main text + appendix)
  make_figures.py          regenerates paper figures from experiments/results/
experiments/
  scripts/                 all experiment + analysis code (entry points below)
  results/                 per-input .jsonl logs + .md reports for every experiment
  logs/                    run logs
  data/                    synthetic NIAH inputs
literature/                landscape survey (topic folders + INDEX.md); 12_iclr_style_exemplars/
external/trimkv/           DBTrimKV (Bui et al.) baseline, with our gated-DBTrimKV drivers
theory_h3.md               working theory notes (SNR / dilution / scaling)
problem_statement.md       problem framing and novelty positioning
```

## Reproducing

Environment: Python 3.13, PyTorch 2.11, Transformers 5.9 (a local `.venv`, not tracked). The
two published head-to-heads use the baselines' own pinned environments (DBTrimKV: the
`external/trimkv` env with Transformers 4.57.1 / PyTorch 2.8 / FlashAttention 2).

All runs use greedy decoding and a fixed seed. Every reported number is reproducible from the
released per-input `.jsonl` in `experiments/results/`; the analysis scripts recompute the
tables directly.

Core gate (single model, RULER 4K mixed suite):

```bash
.venv/bin/python experiments/scripts/gated_eviction.py \
  --model Qwen/Qwen2.5-1.5B-Instruct --config 4096 \
  --tasks niah_multikey_3,vt,fwe,qa_1 --max_examples 100 \
  --budgets 1.0,0.5,0.25,0.125,0.0625 --tau 0.07 --score_policy snapkv \
  --gpu 0 --out experiments/results/gated_4k_qwen15b.jsonl
```

Key entry points (see each script's `--help`):

| Script | Produces |
|---|---|
| `gated_eviction.py` | the gate + base evictors (snapkv/h2o/streamingllm/pyramidkv/manifoldkv) |
| `ruler_sweep.py` | per-task RULER recovery-rate (rho) sweeps |
| `method_agnostic_matrix.py` | the 4×4 base-evictor × model matrix |
| `gate_signal_ablation.py` | D vs. five alternative prefill statistics |
| `pareto_oracle_analysis.py` | accuracy–memory frontier + oracle-gap (analysis only) |
| `matched_memory_analysis.py` | matched-*memory* comparison + crossover |
| `manifoldkv_adakv.py` | faithful ManifoldKV + Ada-KV control |
| `measure_flashattn_gate.py` | two-pass gate deployment cost at scale |
| `calibration_recipe.py` / `normalized_predictor.py` | tau calibration + z-scored transfer |

Rebuild the paper:

```bash
cd paper/iclr2026 && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Limitations

- The gate needs **materialized prefill attention**, incompatible with fused FlashAttention
  serving; it requires a separate attention-exposing pass (cost O(L·H²·k), ~350 ms on a 32B
  model at 4K; not viable for a 70B model at 32K in a paged stack).
- Compression is bounded by the gate-open fraction (≤ ~3×); at aggressive budgets plain eviction
  is on/above the frontier, and a strong trained evictor (DBTrimKV) beats PAGE at matched memory.
- Confirmed capacity-bound exemplars are NIAH-MK3 (synthetic) and `lcc` (realistic); the
  predictor is one-sided (a documented false positive on `passage_count`, a false negative on
  the single-near-tie MK2).
- The theory is a conditional analysis on a single-head surrogate, not a statement about real
  transformers; the empirical results do not depend on it.

## Citation

```bibtex
@misc{mishra2026page,
  title  = {PAGE: Partition-Aware Gated KV-Cache Eviction},
  author = {Mishra, Subhankar},
  year   = {2026},
  note   = {Under review}
}
```
