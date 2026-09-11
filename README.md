# PAGE: Partition-Aware Gated KV-Cache Eviction

A training-free, label-free gate that decides, **per input and before decoding, whether to
evict the KV cache at all**. It computes one scalar from the prefill attention — the
early-to-late-layer drop in cross-layer attention-head agreement — and runs any SnapKV-style
evictor when the drop is large, otherwise keeps the full cache. Read honestly, PAGE is a
**safety mechanism**: it prevents catastrophic accuracy collapse on capacity-bound inputs
(precise multi-key retrieval, exact code completion) at a bounded memory cost, and improves the
accuracy–memory frontier where moderate compression is the target.

> **Status.** Research code for a paper submission. The method is deliberately scoped:
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
- **Prevalence.** In realistic (non-synthetic) traffic the capacity-bound class the gate
  protects is concentrated, not common: ~3.7% of inputs pooled across 9 LongBench subtasks at
  matched budget.
- **Generalization checks.** An architecture-bias control (Llama-3.1-8B) rules out KV-head
  count as the driver of a per-head-allocation anomaly seen on Mistral; a learned probe over
  the richer per-layer profile does not out-predict the simple endpoint statistic `D`; and a
  DynamicKV-style adaptive-budget baseline does not substitute for the gate's per-input
  admission decision.

## Setup

```bash
git clone <repo-url> && cd page-kv
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins `torch==2.11.0` without a CUDA build tag; a bare `pip install` from that
file pulls whatever default build PyPI serves, which is not guaranteed to match your driver. For
a GPU environment, install the CUDA build explicitly instead (adjust `cu128` to your driver's
supported CUDA version):

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt   # the rest of the pins, torch already satisfied
```

Requires Python 3.13 and a CUDA-capable GPU for any GPU experiment (single-card 80GB is enough
for every cell except Qwen2.5-14B at 4K, which needs two cards — see the standing caveats in
`experiments/results/README.md`). The zero-GPU analysis scripts (`experiments/run_all.sh` and
most of `experiments/scripts/`) need only `numpy` and `matplotlib` from `requirements.txt` (no
torch/transformers/datasets), read the released `.jsonl` logs already in `experiments/results/`,
and run on CPU. Wall-clock runtime per GPU cell is not tracked in this checkout — see the
compute-requirements note in `experiments/results/README.md`; VRAM is the documented constraint.

**Hugging Face access.** `meta-llama/Llama-3.1-8B-Instruct` (used by the architecture-bias-control
experiment) is a gated model: accept its license at huggingface.co and authenticate before running
any script that loads it —

```bash
huggingface-cli login          # or: export HF_TOKEN=<your token>
```

Qwen2.5/Qwen3 and `mistralai/Mistral-7B-Instruct-v0.3` are not gated and need no login, though an
`HF_TOKEN` still raises your download rate limit.

The DBTrimKV baseline (`external/trimkv/`) needs a **separate** environment with its own pinned
versions (Transformers 4.57.1 / PyTorch 2.8 / FlashAttention 2); see `external/trimkv/README.md`
and `NOTES.md`.

## Repository layout

```
experiments/
  scripts/                 all experiment + analysis code (entry points below)
  results/                 per-input .jsonl logs + .md reports for every experiment
                            (see experiments/results/README.md for the full index)
  gpu/                      GPU launch scripts for each experiment
  logs/                     run logs (not published; local only)
  data/                     synthetic NIAH inputs
preregistration/            frozen, sha256-hashed predictions for six experiments,
                            verified by the corresponding launch/analysis script at run time
external/trimkv/             DBTrimKV (Bui et al.) baseline, with our gated-DBTrimKV drivers
```

## Reproducing

Environment: Python 3.13, PyTorch 2.11, Transformers 5.9 (a local `.venv`, not tracked). The
DBTrimKV head-to-head uses the baseline's own pinned environment (Transformers 4.57.1 /
PyTorch 2.8 / FlashAttention 2, under `external/trimkv`).

All runs use greedy decoding and a fixed seed. Every reported number is reproducible from the
released per-input `.jsonl` in `experiments/results/`; the analysis scripts recompute the
tables directly. Run the full zero-GPU verification suite with:

```bash
cd experiments && ./run_all.sh     # 11 steps, exits non-zero on any CHECK failure
```

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
| `adakv_matrix.py` / `adakv_analysis.py` | per-head Ada-KV baseline for the headline matrix |
| `prevalence_survey.py` | capacity-bound class prevalence in realistic workloads |
| `profile_probe_ablation.py` | learned per-layer probe vs. the endpoint statistic D |
| `dynamickv_headtohead_analysis.py` | DynamicKV adaptive-budget baseline vs. the gate |

See `experiments/results/README.md` for the complete results index, grouped by what each
experiment settles, with regeneration commands for every group.

## Limitations

- The gate needs **materialized prefill attention**, incompatible with fused FlashAttention
  serving; it requires a separate attention-exposing pass (cost O(L·H²·k), ~350 ms on a 32B
  model at 4K; not viable for a 70B model at 32K in a paged stack).
- Compression is bounded by the gate-open fraction (≤ ~3×); at aggressive budgets plain eviction
  is on/above the frontier, and a strong trained evictor (DBTrimKV) beats PAGE at matched memory.
- Confirmed capacity-bound exemplars are NIAH-MK3 (synthetic) and `lcc` (realistic); the
  predictor is one-sided (a documented false positive on `passage_count`, a false negative on
  the single-near-tie MK2). In realistic traffic the class is a small minority of inputs
  (~3.7% pooled, see `experiments/results/prevalence_survey.md`), not a general property of
  long-context workloads.
- The theory is a conditional analysis on a single-head surrogate, not a statement about real
  transformers; the empirical results do not depend on it.
- A richer learned predictor over the per-layer agreement profile does not fix the fixed-tau
  transfer failure on Llama-family models; the failure is not in the endpoint statistic.

## Citation

```bibtex
@misc{anonymous2026page,
  title  = {PAGE: Partition-Aware Gated KV-Cache Eviction},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review}
}
```
