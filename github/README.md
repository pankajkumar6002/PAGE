# PAGE: Partition-Aware Gated KV-Cache Eviction — anonymized code

Anonymized source code for the ICLR submission *PAGE: Partition-Aware Gated KV-Cache Eviction*.
This supplement contains the method and analysis code needed to reproduce the paper's results.
It has been anonymized for double-blind review (no author, institution, or hosting information).

## Method in one paragraph

PAGE decides, per input and before decoding, **whether** to evict the KV cache at all. It
computes one label-free scalar from the prefill attention — the early-to-late-layer drop in
cross-layer attention-head agreement, `D` — and runs any SnapKV-style evictor when `D >= tau`,
otherwise keeps the full cache. It acts as a safety mechanism: it prevents catastrophic accuracy
collapse on capacity-bound inputs (precise multi-key retrieval, exact code completion) at a
bounded memory cost, and improves the accuracy–memory frontier where moderate compression is the
target. No training, no labels, no fine-tuning.

## Layout

```
experiments/scripts/
  gated_eviction.py           the gate + base evictors (snapkv/h2o/streamingllm/pyramidkv/manifoldkv)
  ruler_sweep.py              per-task RULER recovery-rate (rho) sweeps
  method_agnostic_matrix.py   the 4x4 base-evictor x model matrix
  gate_signal_ablation.py     D vs. five alternative prefill statistics
  pareto_oracle_analysis.py   accuracy-memory frontier + oracle gap (analysis only)
  matched_memory_analysis.py  matched-*memory* comparison + crossover
  manifoldkv_adakv.py         faithful ManifoldKV + Ada-KV control
  measure_flashattn_gate.py   two-pass gate deployment cost at scale
  calibration_recipe.py       one-shot tau calibration
  normalized_predictor.py     per-model z-scored threshold transfer
  build_niah.py               synthetic NIAH input generation
  ... (analysis + aggregation scripts for each experiment)
```

Scripts read/write under `experiments/results/` (relative). Run them from this directory.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Models are the public Hugging Face checkpoints named on each script's command line
(Qwen2.5-1.5B/3B/14B/32B-Instruct, Mistral-7B-Instruct-v0.3, Yi-1.5-9B-Chat,
Llama-3.1-8B-Instruct, Qwen3-4B-Instruct). Tasks use the `simonjegou/ruler` port of RULER and
LongBench. All runs use greedy decoding and a fixed seed.

## Reproducing the core result

The gate on a single model, RULER 4K mixed suite:

```bash
python experiments/scripts/gated_eviction.py \
  --model Qwen/Qwen2.5-1.5B-Instruct --config 4096 \
  --tasks niah_multikey_3,vt,fwe,qa_1 --max_examples 100 \
  --budgets 1.0,0.5,0.25,0.125,0.0625 --tau 0.07 --score_policy snapkv \
  --gpu 0 --out experiments/results/gated_4k_qwen15b.jsonl
```

Each script writes a per-input `.jsonl` (id, task, budget, drop, gate decision, kept-KV, and
plain/gated correctness). The matching analysis scripts recompute the paper's tables from these
files, so every reported number is reproducible from the logs. Run any script with `--help` for
its full argument list.

Notes:
- `gated_eviction.py` supports two-pass prefill (`--two_pass --attn_impl sdpa`) for long
  contexts; the gate signal `D` requires materialized prefill attention (see the paper's
  deployment limitation).
- The DBTrimKV head-to-head uses the DBTrimKV baseline (Bui et al.) run in its own pinned
  environment; that third-party code is not bundled here.

## Scope

PAGE is a moderate-compression (≤ ~3×) safety wrapper. It does not compete in the
aggressive-compression regime and does not beat a strong *trained* evictor at matched memory;
the gate signal needs an attention-exposing prefill pass, which fused-kernel serving does not
provide. These limitations are documented in the paper.
