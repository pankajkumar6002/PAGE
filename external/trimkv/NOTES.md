# TrimKV / DBTrimKV — architecture and integration notes

Source: https://github.com/ngocbh/trimkv (Bui et al., "Cache what lasts: Token
retention for memory-bounded KV cache in LLMs", arXiv 2512.03324, 2025).
Frozen at transformers 4.57.1. Sister paper: "Make Each Token Count"
(arXiv 2605.09649) — same author group, this codebase covers both.

## TL;DR for our project

- **Pretrained gate weights ARE shipped** on Hugging Face for several base
  models (Qwen3-1.7B/4B/8B/14B-Math, Qwen3-4B-Instruct-2507,
  Phi-3-mini-128k). DBTrimKV checkpoints exist for Qwen3-4B-Math and
  Qwen3-4B-Instruct-2507. No Qwen2.5 checkpoints (closest LLM is
  Qwen3, smallest DBTrimKV is 4B).
- **Inference reproduced end-to-end** in a separate venv at
  `external/trimkv/venv` (Python 3.12, torch 2.8.0+cu128, transformers
  4.57.1, flash_attn 2.8.3). See `repro_dbtrimkv.py` and `repro_compare.py`.
- **Cannot reuse our project venv** — they pin transformers==4.57.1, we run
  5.9.0. The TrimKV attention forward depends on private APIs
  (`ALL_ATTENTION_FUNCTIONS`, the 5-tuple cache `.update()` contract, the
  RotaryEmbedding signature, the Qwen3 decoder layer hooks) that all
  changed in transformers 5.x. Their own README warns the code is
  "frozen at a version close to what produced the paper results".
- **DBTrimKV is not a drop-in policy** like SnapKV/H2O — it's a *whole
  modified attention module* with a learned gating MLP per layer. To wrap
  it with our gating, we'd integrate at the **PagedTrimKVCache compress()
  step**, not at attention.

## Repo layout

```
src/trimkv/
  cache_utils.py        TrimKVCache, DynamicBudgetTrimKVCache,
                        PagedCache, PagedTrimKVCache (~1100 LoC)
  attn/
    eager_attn.py       retention-gated attention (eager)
    flex_attn.py        retention-gated attention (FlexAttention, training)
    db_flash_attn.py    dynamic-budget + paged flash attention 2 (inference)
    __init__.py         ALL_ATTENTION_FUNCTIONS registry
  triton/
    db_cache_utils.py        paged-cache index ops
    retention_sum.py         triton kernel for the log-G prefix sum
    retention_sum_packed.py  packed variant
  models/{qwen3, qwen2, llama, phi3, qwen2_5_vl, qwen3_vl, llava}/
    configuration_trimkv_*.py
    modeling_trimkv_*.py     adds RetentionGate(10) module + replaces
                             attention with TrimKV*Attention
train/llm/                   DeepSpeed + Trainer recipe for gate training
                             (frozen base model, only gate params trainable)
train/vlm/                   same harness for VLMs
experiments/{scbench, longbench, longbench_v2, math, longproc,
             longmemeval, mmdu, benchmarks, lmms-eval}/
```

## Architecture: how DBTrimKV works

### Training (offline, recipe in `train/llm/`)

- Base LLM weights are **frozen**. Only the retention gate (and an
  optional rope-bias) is trainable (`trainable_params="self_attn.retention_gate"`).
- For each KV head, a small MLP (`RetentionGate10`: 3 linear layers,
  hidden_size → 512 → 512 → num_kv_heads, with SiLU and a learnable
  bias initialized to `+18.0` so that initial gates are open) emits a
  per-token, per-head **retention score** in `log σ` form.
- Loss is forward KL of the next-token distribution between the
  full-cache teacher and the trimmed-cache student. The R1-style math
  recipe trains on OpenR1-Math-220k, the long-context recipe on
  synth_long/booksum/buddhi. See `train/llm/README.md`.
- Two env vars switch TrimKV vs DBTrimKV during training:
  `RETENTION_GATE={rg, rg10}`, `GLOBAL_CAPACITY={False, True}`. Same loss
  surface, same data.
- DBTrimKV-Qwen3-4B-Math was trained at 32k ctx, `memory_size=128`,
  `retention_gate_bias_init=18.0`.

### Inference (`src/trimkv/cache_utils.py`, `src/trimkv/attn/db_flash_attn.py`)

At every decoding step:

1. The standard Qwen attention runs `q_proj/k_proj/v_proj/q_norm/k_norm`
   and RoPE.
2. **Before** appending the new KV to the cache, the
   `RetentionGate10` reads `hidden_states` (pre-RoPE) and emits
   `retention_weights` of shape `(B, num_kv_heads, S)` in `log σ` space.
3. `PagedTrimKVCache.update(k, v, retention_weights)` writes the new
   tokens into a paged buffer (256-token blocks). When total tokens
   exceed `memory_size × num_layers × num_kv_heads` (the *global*
   budget), `PagedTrimKVCache.compress()` evicts the lowest-score
   tokens **per head, but redistributes blocks dynamically** so heads
   under retention pressure can borrow from heads with slack. The
   eviction score for a token at age `t-i` is the retention weight
   decayed by `compute_log_G(log_beta=retention_weight, t, i, n)` —
   the closed-form exponential decay derived in the paper.
4. Attention uses `paged_flash_attention_2` (their fork of
   `flash_attn_with_kvcache`) over the paged cache.

Buffer + recent window (`buffer_size=32`) are always kept verbatim;
only history older than the buffer is subject to eviction.

### Key cache classes

- `TrimKVCache`: per-layer fixed-size buffer; basic TrimKV inference path.
- `DynamicBudgetTrimKVCache`: per-layer dynamic budget but flat tensors.
- `PagedTrimKVCache` **(this is DBTrimKV's runtime)**: paged-attention
  blocks (size 256) in a global pool; physical pool grows on demand
  (we observed doublings 576 → 1152 → 2304 → 4608 blocks on a
  4.5k-token prompt). Each `(layer, head)` has its own block table
  into the pool. Compression happens layer-by-layer after `update`.

### `model.generate` integration

`TrimKVQwen3ForCausalLM` overrides `from_pretrained` to:
1. Snapshot-download the HF repo (`config.json` + `trimkv_weights.pth`,
   no base-model weights — those are fetched from `config.base_model`).
2. Load the base model with TrimKV config.
3. `model.load_state_dict(gate_weights, strict=False)` to overlay the
   retention-gate parameters.

`model.generate(..., past_key_values=PagedTrimKVCache(...))` then runs
the standard HF generation loop, with each attention layer pulling
the next 5-tuple `(k, v, retention_weights, kv_positions,
flash_attn_kwargs)` from the cache. There is no `vanilla_forward`
codepath exposed at the `generate` API, so a "TrimKV model with full
KV cache" baseline requires a code patch.

## Public checkpoints (Hugging Face)

LLM:
- `ngocbh/TrimKV-Qwen3-1.7B-Math`, `-4B-Math`, `-8B-Math`, `-14B-Math`
- `ngocbh/TrimKV-Qwen3-4B-Instruct-2507`
- `ngocbh/TrimKV-Phi-3-mini-128k-instruct`
- `ngocbh/DBTrimKV-Qwen3-4B-Math` — smallest DBTrimKV LLM
- `ngocbh/DBTrimKV-Qwen3-4B-Instruct-2507`

VLM (DBTrimKV only): `DBTrimKV-Qwen3-VL-8B-Thinking`,
`DBTrimKV-Qwen3-VL-4B-Instruct`.

**No Qwen2.5-1.5B / 3B / 7B-Instruct checkpoints** for either TrimKV or
DBTrimKV. The closest small model we can run is `Qwen3-1.7B-Math` for
TrimKV; **for DBTrimKV the only option is 4B+**.

Checkpoint contents (`DBTrimKV-Qwen3-4B-Math`):
- `config.json` — extends Qwen3Config with TrimKV fields
  (`retention_gate=rg10`, `global_capacity=true`, `memory_size=128`,
  `retention_gate_bias_init=18.0`, `retention_gate_intermediate_size=512`,
  `tie_retention_gate_layers=true`).
- `trimkv_weights.pth` — 113 MB, 216 tensors. Per-layer `linear1/2/3
  + bias` for all 36 layers, no base-model weights.
- `README.md`.

Despite `tie_retention_gate_layers=true`, the shipped file has distinct
weights per layer (we verified this) — the "tied" terminology in the
README refers to the **architectural parameterisation** (the final
projection structure is shared across heads within a layer), not literal
weight sharing across layers.

## Supported models / datasets out of the box

Models (with `TrimKV*ForCausalLM` wrapper classes shipped):
- Qwen3, Qwen2, Llama, Phi-3, Qwen2.5-VL, Qwen3-VL, LLaVA.
- **No native Qwen2.5 text-only support** (only Qwen2 and Qwen3).

Datasets (training; under `train/llm/dataset/`):
`openr1_math`, `ultrachat`, `synth_long`, `long_alpaca`, `buddhi`,
`booksum`, `niah`, `prolong`.

Evaluation benchmarks (under `experiments/`):
- SCBench (Microsoft) — 8 tasks: `scbench_kv` (the closest analogue to
  RULER NIAH-MK3), `scbench_vt`, `scbench_qa_eng`, `scbench_choice_eng`,
  `scbench_summary`, `scbench_mf`, `scbench_summary_with_needles`,
  `scbench_repoqa`.
- LongBench, LongBench v2, LongMemEval, LongProc, MMDU, lmms-eval, and
  custom math (GSM8K, MATH-500, AIME-24).
- **No RULER NIAH-MK3 harness in the repo.** Closest in-repo task is
  `scbench_kv` (key-value retrieval, structurally identical to RULER
  NIAH-MK3) and the `niah` training dataset loader.

## Reproduction in this directory

### What we did (working venv at `external/trimkv/venv`)

```bash
# Built from scratch — does NOT reuse our project's .venv
uv venv --python 3.12 venv
uv pip install --python venv/bin/python torch==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128
uv pip install --python venv/bin/python transformers==4.57.1 \
    accelerate einops numpy huggingface_hub ninja packaging wheel
uv pip install --python venv/bin/python -e .
uv pip install --python venv/bin/python "flash-attn>=2.7.2" \
    --no-build-isolation       # used 2.8.3
```

Total install time ~3 min (flash-attn was a prebuilt wheel for
torch 2.8.0+cu12.8/Py3.12, no source build required).

### Smoke test: `repro_dbtrimkv.py`

Single-example DBTrimKV inference on a 32-pair UUID→UUID NIAH-style
prompt (2287 input tokens), DBTrimKV-Qwen3-4B-Math, `memory_size=128`:

- Wall time: 3.35 s for 27 generated tokens (8.1 tok/s, single A100)
- Peak GPU memory: **9.21 GB** (the 4B model in bf16 alone is ~7.8 GB)
- Peak cached tokens across all layers/heads: **658,656**
- Cache pool grew 576 → 1152 → 2304 → 4608 blocks dynamically
- Quality: output first 11 chars of the target UUID are correct
  (`489be078-86c`), suffix hallucinated. Expected behavior for a
  4B math-tuned model on an OOD UUID-retrieval prompt at 2K context.

### Comparison run: `repro_compare.py`

Same flow, 4451-token prompt, two budgets:

| Cache              | Time   | tok/s | Peak GPU | Output                      | Match  |
| ------------------ | ------ | ----- | -------- | --------------------------- | ------ |
| DBTrimKV M=128     | 4.07s  | 7.9   | 10.23 GB | first 11 chars correct      | prefix |
| DBTrimKV M=512     | 3.49s  | 9.7   | 11.50 GB | exact target UUID retrieved | exact  |

(A "TrimKV model with full DynamicCache" baseline is not available
without patching the TrimKV attention to skip the 5-tuple unpack —
their `Cache.update` contract is non-standard.)

The exact-match success at M=512 + prefix-match at M=128 confirms the
gate is actually loaded and applied. The pool-growth log lines confirm
the paged dynamic-budget mechanism is active.

## Feasibility for gated-DBTrimKV (ICLR experiment)

### Verdict: **partial — requires modest engineering, NOT infeasible**

Pretrained gates ARE shipped, so we do not need to retrain on 4×A100.
The 4B-Math model is the only DBTrimKV LLM under 8B. If our gating
wrapper at 4B-Math improves over their numbers, that is a strong
headline result.

### What integration would actually look like

Three layers of work, roughly in order:

1. **Run their baseline numbers** (effort: ~1 day).
   Add a runner inside `experiments/scbench/run_scbench.py` (or our
   wrapper) that wraps DBTrimKV-Qwen3-4B-Math inference at our chosen
   budgets and reports per-task scores. SCBench-KV / SCBench-VT cover
   the NIAH-style retrieval our gating targets. This produces the
   "baseline DBTrimKV" number our gated version has to beat.

2. **Wire our gating wrapper into PagedTrimKVCache.compress()**
   (effort: ~1-2 days).
   Our gating wrapper is a scoring policy that modulates retention
   scores per (layer, head, token). The clean injection point is
   `PagedTrimKVCache.compress()` (cache_utils.py:884-1035), which
   currently scores tokens using `retention_weights + log_G(beta, t, i,
   n)`. We replace that with `gating_wrapper(retention_weights,
   log_G, k, v, q_state) → modulated_scores`. Their score is in
   log-space so additive composition is natural. Per the task brief,
   we already have such a wrapper for SnapKV; the same `score_policy`
   interface plugs in here with one new policy adapter.

3. **Run gated-DBTrimKV sweep across budgets** (effort: ~1 day, 4×A100).
   At memory_size ∈ {64, 128, 256, 512} on SCBench's 8 tasks, ~20 GB/GPU,
   ~30 min/task at 4B parameters. Comfortably fits in a few hours.

Total estimate: **3-4 engineer-days + 1 GPU-day**, assuming our gating
wrapper API is already stable (which it is per the task list — #35/#36
are completed).

### Non-trivial risks / unknowns

- **Two transformers versions in the same project.** We must keep
  the TrimKV venv separate from our main venv. Eval results must be
  written to disk as JSONL and aggregated externally — no in-process
  comparison.
- **Closed-form retention decay vs. our gating.** Their `log_G` decay
  is parametrically baked in. Whether our gate composes additively or
  multiplicatively with the *learned* `retention_weights` is a design
  choice that needs a quick ablation. (My read: additive in log space
  is the right starting point because both are already log-probability
  modulations.)
- **Only 4B model.** If we want to show generality (gating helps at
  multiple scales), we'd have to either (a) train our own DBTrimKV
  gate on Qwen3-1.7B-Math (their training recipe is published, 4×A100
  for ~1-2 days at 32k context with stage2 deepspeed; expensive but
  feasible), or (b) accept "single-scale SOTA win" framing.
- **No RULER NIAH-MK3 in their repo.** We'd need to either port our
  RULER harness against their model loader (preferred — small lift)
  or use SCBench-KV as the structurally equivalent task and note the
  benchmark substitution in the paper.
- **The `vanilla_forward` flag isn't exposed at `generate`.** If we
  want a "no-gate" ablation of the same 4B model, we either patch the
  `model.generate` call path or load Qwen3-4B base separately.

## Bottom line for the parent task

- ✅ **Code cloned and documented.**
- ✅ **Architecture mapped** (RetentionGate10 MLP per layer, paged
  global-budget cache, `paged_flash_attention_2` runtime).
- ✅ **Pretrained gate weights confirmed shipped** for DBTrimKV-Qwen3-4B
  (113 MB, 216 tensors, all retention-gate params).
- ✅ **Inference reproduced** end-to-end on a NIAH-style prompt. Exact
  UUID retrieval at M=512, ~8-10 tok/s, ~10 GB peak.
- ⚠️ **Smallest DBTrimKV LLM is 4B** (no Qwen2.5-1.5B option).
- ⚠️ **Their repo has no RULER NIAH-MK3 harness** — closest in-repo
  task is SCBench-KV (structurally identical).
- ✅ **Integration is feasible** in 3-4 days + ~1 GPU-day, gated on
  injecting our scoring policy into `PagedTrimKVCache.compress()` and
  running their SCBench harness (or porting our RULER harness against
  their model wrapper).
