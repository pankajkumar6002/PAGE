# MMDU evaluation

This directory evaluates TrimKV / DBTrimKV and a set of decode-time KV-compression baselines (R-KV, SnapKV, AdaKV, AdaPyramidKV) on the [MMDU](https://huggingface.co/datasets/laolao77/MMDU) benchmark — multi-turn multi-image dialog understanding (Liu et al., NeurIPS 2024).

The eval pipeline is two-stage: `run_mmdu.py` generates dialog responses per (method, KV-budget) configuration and writes JSON outputs, then `run_eval_mmdu.py` scores them with a judge LLM (Gemini or a local Qwen3-VL judge served via vLLM / transformers) using the rubric in [meta_prompt.txt](meta_prompt.txt).

## Layout

- [run_mmdu.py](run_mmdu.py) — generation entry point. Loads the chosen method via `load_model.py::LOADER_MAP` and writes `results/<run_name>.json` per config.
- [run_eval_mmdu.py](run_eval_mmdu.py) — judge-LLM scoring entry point. Reads a generated `*.json`, calls Gemini or a local judge model, and writes `<run_name>_scored.json` plus an aggregated metrics file.
- [run_eval_mmdu_gemini.py](run_eval_mmdu_gemini.py) — Gemini-only scoring variant (kept for compatibility with older runs).
- [load_model.py](load_model.py) — `LOADER_MAP` registry: `vanilla`, `trimkv`, `dbtrimkv`, `snapkv`, `rkv`, `adakv`, `adapyramidkv`.
- [meta_prompt.txt](meta_prompt.txt) — judge rubric (six dimensions: Creativity, Richness, Visual Perception, Logical Coherence, Answer Accuracy, Image Relationship Understanding; each scored 1–10, plus an Overall Score).
- [utils.py](utils.py) — shared helpers; `load_dataset(config)` auto-downloads `laolao77/MMDU` (`benchmark.json` + `mmdu_pics.zip`) into `config.dataset_dir` on first run.
- [baselines/](baselines/) — vendored R-KV / SnapKV / AdaKV compression policies and the corresponding `transformers` monkey-patches.
- [scripts/](scripts/) — `eval_dbtrimkv.sh`, `eval_baselines.sh`, `debug.sh` (smoke test), `eval_outputs.sh` (sweep-score a results folder).
- `data/` — populated on first run by `utils.load_dataset`. Gitignored.

## Setup

From the repo root:

```bash
pip install -e .
pip install -r experiments/mmdu/requirements.txt   # if present, otherwise use the project requirements
```

For the **judge LLM** during scoring, set one of:

```bash
export GEMINI_API_KEY=...           # required for `--inference_backend gemini` (default)
# or run a local Qwen3-VL judge with vLLM / transformers — see scripts/eval_outputs.sh
```

The MMDU benchmark itself (`benchmark.json`, ~3.6 MB; `mmdu_pics.zip`, ~143 MB extracted) is fetched automatically from `laolao77/MMDU` the first time `run_mmdu.py` or `run_eval_mmdu.py` is run.

## Generation

### DBTrimKV (default)

```bash
MODEL_PATH=ngocbh/DBTrimKV-Qwen3-VL-4B-Instruct \
DOWNLOAD_FROM=huggingface \
bash scripts/eval_dbtrimkv.sh
```

Uses the global-capacity dynamic-budget variant. The 4B-Instruct and 8B-Thinking checkpoints are both on HF — see [the project model registry](../../README.md#vlm-checkpoints).

### Baselines

```bash
MODEL_PATH=Qwen/Qwen3-VL-4B-Instruct \
METHODS=snapkv,adakv,rkv \
bash scripts/eval_baselines.sh
```

The script handles the `vanilla` no-compression case specially — when `METHODS=vanilla`, the budget loop terminates after one run (no KV budget to sweep). Add `adapyramidkv` to the comma-separated list to run that variant.

### Smoke test

For a 4-sample sanity check on a single budget:

```bash
bash scripts/debug.sh
```

Override via `MODEL_PATH=`, `METHODS=`, `BUDGET=`, `N_SAMPLES=`.

### Direct invocation

```bash
python run_mmdu.py \
    --model_path ngocbh/DBTrimKV-Qwen3-VL-4B-Instruct \
    --method dbtrimkv \
    --download_from huggingface \
    --kv_budget 256 \
    --max_new_tokens 4096 \
    --n_samples 10
```

Useful flags: `--start_idx`/`--end_idx` (partial runs), `--rerun` (overwrite existing output JSONs), `--disable_thinking` (turn off `<think>` tokens for thinking-mode VLMs), `--attn_implementation` (`flash_attention_2` by default).

### Slurm

`LAUNCHER=slurm` (or `slurm_nmi`) submits each per-(method, budget) run via `sbatch scripts/wrapper*.sh python …`. The wrapper scripts are local-only (gitignored); supply your own that match your cluster.

## Scoring

After generations land under `results/`, sweep-score them with:

```bash
GEMINI_API_KEY=... bash scripts/eval_outputs.sh path/to/results_dir/
```

This wraps `run_eval_mmdu.py` over every `<method>-<budget>b-<length>l-<max_new_tokens>t.json` file in the folder. Each scored output gets a `*_scored.json` next to it; the script reports per-dimension and overall scores using the rubric in [meta_prompt.txt](meta_prompt.txt).

To use a local judge model instead of Gemini, pass `--inference_backend transformers` (and optionally `--model_name <hf_id>` to override the default `Qwen/Qwen3-VL-32B-Instruct`).

## Acknowledgements

- [MMDU](https://github.com/Liuziyu77/MMDU) (Liu et al., NeurIPS 2024) — the benchmark, the `mmdu_pics` images, and the judge rubric we adapt.
- [R-KV](https://github.com/Zefan-Cai/R-KV) — the SnapKV / R-KV / StreamingLLM / H2O compression policies vendored under [baselines/rkv/](baselines/rkv/).
- [AdaKV](https://github.com/FFY0/AdaKV) — head-level adaptive budget vendored inside [baselines/rkv/](baselines/rkv/).
