# lmms-eval — TrimKV multimodal benchmarks

This directory contains the evaluation harness used to benchmark TrimKV / DBTrimKV (and KV-compression baselines: R-KV, SnapKV, AdaKV, AdaPyramidKV) on a suite of image and video multimodal tasks. It's a fork of [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) extended with TrimKV model loaders and the R-KV-style baseline tree under [rkv/](rkv/).

## Layout

- [run_benchmark.py](run_benchmark.py) — entry point. Wires a model, a compression method, and one or more lmms-eval tasks together.
- [load_model.py](load_model.py) — loader registry mapping `--method` (`vanilla`, `trimkv`, `dbtrimkv`, `rkv`, `snapkv`, `adakv`, `adapyramidkv`) to a model-construction function.
- [lmms_eval/](lmms_eval/) — the upstream evaluation library (tasks, evaluator, loggers, model wrappers).
- [rkv/](rkv/) — TrimKV's vendored R-KV / SnapKV / AdaKV compression policies and the monkey-patches that inject them into Qwen2.5-VL / Qwen3-VL.
- [scripts/](scripts/) — `eval_dbtrimkv.sh`, `eval_text_baselines.sh`, plus local-only Slurm `wrapper*.sh` (gitignored).
- `results/`, `logs/`, `artifacts/` — per-run outputs (gitignored).

## Setup

Install the (forked) `lmms_eval` package and the Qwen-VL helper utilities:

```bash
cd experiments/lmms-eval
pip install -e .
pip install qwen_vl_utils
```

Copy the shared `.env` (or create your own) — it must define `OUTPUT_DIR`, `DATASET_DIR`, `HF_TOKEN`, and the `WANDB_*` variables used by the loggers:

```bash
cp ../../.env ./
```

## Datasets

Most lmms-eval tasks fetch their own datasets through 🤗 Hub on first use. For the video benchmarks, pre-download them so distributed workers don't hit rate limits:

```bash
hf download lmms-lab/VideoMMMU  --repo-type dataset
hf download lmms-lab/Video-MME  --repo-type dataset
hf download lmms-lab/YouCook2   --repo-type dataset
```

## Running evaluations

All scripts in [scripts/](scripts/) accept a `LAUNCHER` env var:

- `LAUNCHER=python` (default) — run inline on the current machine.
- `LAUNCHER=slurm`, `LAUNCHER=slurm_h100`, `LAUNCHER=slurm_a40` — submit via the matching `wrapper_qos*.sh` Slurm wrapper. The wrappers are gitignored; supply your own that match your cluster.

### Default 7-task sweep

The shipped `eval_dbtrimkv.sh` is configured for the seven multimodal tasks reported in the paper:

```
mathvision_testmini
video_mmmu_adaptation
mmmu_pro_vision
videomme
video_mmmu_comprehension
videomathqa_mcq
mmstar
```

Uncomment the multi-dataset line at the top of [scripts/eval_dbtrimkv.sh](scripts/eval_dbtrimkv.sh) to run the full sweep; the default ships with `datasets=(mathvision_testmini)` for a fast smoke test.

### TrimKV / DBTrimKV

```bash
MODEL=ngocbh/DBTrimKV-Qwen3-VL-8B-Thinking \
DOWNLOAD_FROM=huggingface \
METHODS=dbtrimkv \
bash scripts/eval_dbtrimkv.sh
```

`DOWNLOAD_FROM` accepts `huggingface`, `wandb`, or `local`. Pass `METHODS=trimkv` (and the matching `ngocbh/TrimKV-Qwen3-VL-8B-Thinking` model) to use the per-head local-budget variant. Sweeps over `budgets` and `datasets` are configured at the top of the script.

### Baselines (vanilla, SnapKV, R-KV, AdaKV, AdaPyramidKV)

```bash
METHODS=vanilla bash scripts/eval_text_baselines.sh
METHODS=snapkv,rkv,adakv BATCH_SIZE=1 bash scripts/eval_text_baselines.sh
```

Override `MODEL_PATH` to evaluate a different base model (default: `Qwen/Qwen3-VL-8B-Thinking`). Note that `adakv` / `adapyramidkv` force `batch_size=1` internally.

### Direct invocation

For one-off configurations, call `run_benchmark.py` yourself:

```bash
python run_benchmark.py \
    --model ngocbh/DBTrimKV-Qwen3-VL-8B-Thinking \
    --method dbtrimkv \
    --compress_args=kv_budget=128,download_from=huggingface,fixed_kv_budget=True \
    --tasks mathvision_testmini \
    --batch_size 32 \
    --gen_kwargs=max_new_tokens=32768 \
    --output_path ./results/manual/qwen3vl/dbtrimkv \
    --log_samples
```

See `CompressionConfig` in [run_benchmark.py](run_benchmark.py) for the full surface of `--compress_args` fields (KV budget, buffer size, strategy, R-KV window/lambda, …).

## Results

Per-run JSON metrics and sample logs are written under `results/<exp>/<model>/<method>/`. Aggregated summaries are also pushed to W&B when the `WANDB_*` env vars are set.

## Acknowledgements

- [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) — the underlying evaluation harness; we vendor a fork under [lmms_eval/](lmms_eval/).
- [R-KV](https://github.com/Zefan-Cai/R-KV) — the SnapKV / R-KV / StreamingLLM / H2O policies vendored under [rkv/](rkv/).
- [AdaKV](https://github.com/FFY0/AdaKV) — head-level adaptive KV budget vendored inside [rkv/](rkv/).
