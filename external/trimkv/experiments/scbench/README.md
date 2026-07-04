# SCBench evaluation

This directory evaluates **TrimKV** and a set of decode-time KV-compression baselines (R-KV, SnapKV, StreamingLLM, H2O) on [SCBench](https://huggingface.co/datasets/microsoft/SCBench) — Microsoft's KV-cache-centric long-context benchmark covering string retrieval, semantic retrieval, global information processing, and multi-tasking across multi-turn and multi-request modes (Li et al., arXiv 2412.10319).

> **DBTrimKV is not yet supported here.** Only fixed-budget TrimKV is wired up against SCBench's chunked-prefill flow at the moment.

The harness itself is adapted from [microsoft/MInference/scbench](https://github.com/microsoft/MInference/tree/main/scbench); we reuse `eval_utils.py`, `compute_scores.py`, and `repo_qa_utils.py` essentially as-is and replace the model-loading layer with our own [load_model.py](load_model.py) so we can plug in TrimKV.

## Layout

- [run_scbench.py](run_scbench.py) — entry point. Loads the chosen method via `load_model.py::LOADER_MAP`, runs the SCBench multi-turn / scdq flow, and writes per-task JSONL results to `results/`.
- [load_model.py](load_model.py) — `LOADER_MAP` registry: `fullkv`, `trimkv`, `snapkv`, `rkv`, `streamingllm`, `h2o`, `seerattn`. (`dbtrimkv` is registered but **not validated** against SCBench yet — don't use it.)
- [eval_utils.py](eval_utils.py), [compute_scores.py](compute_scores.py), [repo_qa_utils.py](repo_qa_utils.py) — vendored from MInference SCBench. Provides the multi-turn / shared-context-different-query (`scdq`) prompt builders, the per-task scoring functions, and the `scbench_repoqa` retrieval helpers.
- [rkv/](rkv/) — vendored R-KV / SnapKV / StreamingLLM / H2O / KeyDiff compression policies and their `transformers` monkey-patches.
- [scripts/](scripts/) — `run_trimkv.sh`, `run_baselines.sh`. (`run_dbtrimkv.sh` is a stub — DBTrimKV isn't supported yet, see the note above.)
- [setup/setup_kivi.sh](setup/setup_kivi.sh) — optional KIVI install helper, kept from upstream MInference.
- [cache_blend.yaml](cache_blend.yaml) — upstream MInference KV-blending config; unused in our default flow.

`results/`, `logs/`, `artifacts/` are gitignored.

## Setup

From the repo root:

```bash
pip install -e .
pip install -r experiments/scbench/requirements.txt
```

SCBench data is fetched on-the-fly via `datasets.load_dataset("microsoft/SCBench", <subset>, split="test")`, so no manual download is required (set `HF_TOKEN` if you hit rate limits).

## Running

### TrimKV

```bash
MODEL=ngocbh/TrimKV-Qwen3-4B-Instruct-2507 \
DOWNFROM=huggingface \
METHOD=trimkv \
KV_BUDGET=4096 \
bash scripts/run_trimkv.sh
```

The script sweeps the six tasks reported in our paper:

```
scbench_vt
scbench_qa_eng
scbench_choice_eng
scbench_summary
scbench_mf
scbench_summary_with_needles
```

`MODE=scdq` (the shared-context-different-query mode) is the default; pass `MODE=multiturn` (or comma-separated `MODE=scdq,multiturn`) to run the other shared-context mode.

### Baselines

```bash
MODEL=Qwen/Qwen3-4B-Instruct-2507 \
KV_BUDGET=4096 \
MODE=scdq \
bash scripts/run_baselines.sh
```

The shipped script iterates `METHODS=(fullkv h2o snapkv streamingllm)` over `TASKS=(scbench_repoqa)`. Uncomment the longer task list near the top of [scripts/run_baselines.sh](scripts/run_baselines.sh) to run the full 8-task sweep.

### Direct invocation

```bash
python run_scbench.py \
    --task scbench_vt \
    --model_path ngocbh/TrimKV-Qwen3-4B-Instruct-2507 \
    --method trimkv \
    --download_from huggingface \
    --kv_budget 4096 \
    --max_model_len 128000 \
    --eval_mode scdq
```

Useful flags: `--num_eval_examples N` (truncate dataset for fast iteration), `--max_turns N` (cap multi-turn depth), `--name_suffix tag` (label the output file), `--rewrite` (overwrite existing JSONL).

### Slurm

`LAUNCHER=slurm` (or `slurm_qos`) submits each per-task run via `sbatch scripts/wrapper_resub*.sh python …`. The wrapper scripts are local-only (gitignored); supply your own that match your cluster.

## Results

Per-run JSONLs land under `results/<task>/<run_name>.jsonl`. Aggregate scores are computed inline by `compute_scores.py` at the end of each run and printed to stdout / saved alongside the JSONL.

## Known caveats

We discovered a bug in the upstream R-KV `DynamicCache`: `past_key_values.get_seq_length()` returns the length of the **currently cached** tokens rather than the total tokens seen so far, which produces wrong `cache_position` values during decoding and degrades attention. We use a fixed version of `DynamicCache` for the SCBench experiments reported in our paper.

The math results elsewhere in this repo were collected with the original R-KV `DynamicCache` (i.e. before the fix). If you re-run those with the patched cache, expect small numerical differences from the paper. We do not currently have bandwidth to rerun all baseline configurations; treat the R-KV / SnapKV combinations as best-effort.

## Acknowledgements

- [SCBench](https://github.com/microsoft/MInference/tree/main/scbench) (Li et al., arXiv 2412.10319) — the benchmark, the multi-turn / scdq prompt builders, and the per-task scorers we reuse.
- [R-KV](https://github.com/Zefan-Cai/R-KV) — the SnapKV / R-KV / StreamingLLM / H2O policies vendored under [rkv/](rkv/).
