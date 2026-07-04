# train/vlm — TrimKV training for vision-language models

This directory trains the TrimKV (and DBTrimKV) retention-gate on top of a frozen vision-language model — Qwen2.5-VL, Qwen3-VL, or LLaVA-1.5 — using DeepSpeed + 🤗 Trainer. Only the gating parameters are updated; the visual encoder, projector, and base LLM stay frozen, which keeps memory low even with packed long-form video samples.

## TrimKV vs DBTrimKV

Both variants share the same training loop, datasets, and loss surface. They differ only in **how the KV budget is allocated** and **which retention-gate parameterisation is used**, controlled by two env vars when launching the training script:

| | TrimKV | DBTrimKV |
|---|---|---|
| `RETENTION_GATE` | `rg` | `rg10` |
| `GLOBAL_CAPACITY` | `False` | `True` |
| `TIE_RG_WEIGHTS` | `False` | `True` |
| Budget semantics | per-layer, per-head local budget `M_local = MEMORY_SIZE` | single global budget `M_global = MEMORY_SIZE × num_layers × num_heads`, redistributed dynamically across layers/heads |
| Gate parameterisation | independent retention gate per head | final projection of the gate **tied across layers and heads** |
| Typical `MEMORY_SIZE` | 256 | 32 |

Everything else (`base_model`, `dataset_use`, `model_max_length`, `base_loss`, `lr`, …) is orthogonal.

## Layout

- [train.py](train.py) — entry point. Wraps the base VL model in the matching `TrimKV*ForConditionalGeneration` class and trains the parameters selected by `--trainable_params`.
- [trainer.py](trainer.py) — the `TrimKVVLTrainer` (KL distillation against a vanilla forward pass when `base_loss` ∋ `fwkl` / `rvkl`).
- [argument.py](argument.py) — `ModelArguments`, `DataArguments`, `TrainingArguments` dataclasses; the canonical reference for every CLI flag.
- [dataset/](dataset/) — dataset loaders (`data_processor.py`), per-model preprocessing (`qwen_utils.py`, `llava_utils.py`), the `DATA_CONFIGS` registry in [dataset/configs.py](dataset/configs.py), and packing utilities.
- [chat_template/templates.json](chat_template/templates.json) — overrides for tokenizers whose default chat template strips `<think>…</think>` content.
- [ds_config/](ds_config/) — DeepSpeed configs (`zero2.json` is the supported default; `zero3.json` and `zero3_offload.json` are available for larger setups).
- [scripts/](scripts/) — training launchers (`train_trimkv.sh`, `train_trimkv_mmdu.sh`).
- [scripts/data/](scripts/data/) — **all dataset download & preprocessing recipes** (see [scripts/data/README.md](scripts/data/README.md)).
- [precompute_seqlen.py](precompute_seqlen.py) — caches per-sample token lengths to speed up the packing dataloader.

## Setup

### 1. Environment

Create `train/vlm/.env`. The launch scripts `export $(cat .env | xargs)` at the top, so anything you put here propagates to the run:

```env
OUTPUT_DIR=/abs/path/to/outputs
DATASET_DIR=/abs/path/to/data
HF_HOME=/abs/path/to/hf_cache       # optional
HF_TOKEN=...                        # required for gated datasets
WANDB_MODE=online
WANDB_PROJECT=vlm-finetune
WANDB_ENTITY=your_wandb_account
WANDB_API_KEY=...
```

### 2. Packages

From the repo root:

```bash
pip install -e .
conda install -y -c nvidia cuda-toolkit=12.8
pip install flash-attn --no-build-isolation
```

For the lmms-eval benchmarks (also pulls Qwen-VL helpers):

```bash
pip install -e experiments/lmms-eval/
```

For video benchmarks (`torchcodec` needs an ffmpeg ≥6 ABI):

```bash
conda install -y "ffmpeg<7,>6" -c conda-forge
pip install torchcodec==0.7
```

### 3. Datasets

All download and preprocessing scripts live under [scripts/data/](scripts/data/). Follow [scripts/data/README.md](scripts/data/README.md) for per-dataset recipes — the supported datasets are:

- [Fancy-MLLM/R1-Onevision](https://huggingface.co/datasets/Fancy-MLLM/R1-Onevision) — image reasoning.
- [lmms-lab/M4-Instruct-Data](https://huggingface.co/datasets/lmms-lab/M4-Instruct-Data) — image instruction tuning.
- [lmms-lab/LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K) — `0_30_s_academic_v0_1` subset (open-ended QA + captioning).
- [laolao77/MMDU](https://huggingface.co/datasets/laolao77/MMDU) — multi-image dialog.
- [open-r1/OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) — math reasoning.

After downloading, run [precompute_seqlen.py](precompute_seqlen.py) on each dataset key in `dataset/configs.py::DATA_CONFIGS` to cache sequence lengths. Annotations and image/video roots are resolved relative to `DATASET_DIR` via the entries in `dataset/configs.py` — edit them there if your layout differs.

The default training mixture (set by `DATASETS=` in `scripts/train_trimkv.sh`) is:

```
r1_onevision%30, m4_instruct50_images%40,
academic_openended%30, academic_caption%30,
mmdu_45k%50, math_220k%20
```

`name%N` means "use N% of `name`".

## Training

### Default mixture (DBTrimKV)

The shipped defaults of `scripts/train_trimkv.sh` train **DBTrimKV** (`RETENTION_GATE=rg10`, `GLOBAL_CAPACITY=True`, `TIE_RG_WEIGHTS=True`, `MEMORY_SIZE=32`) on the full mixture above:

```bash
GPUS=8 BASE_MODEL=Qwen/Qwen3-VL-8B-Thinking \
bash scripts/train_trimkv.sh
```

### Train TrimKV (per-head local budget)

Flip the three flags that distinguish the variants:

```bash
GPUS=8 BASE_MODEL=Qwen/Qwen3-VL-8B-Thinking \
RETENTION_GATE=rg \
GLOBAL_CAPACITY=False \
TIE_RG_WEIGHTS=False \
MEMORY_SIZE=256 \
bash scripts/train_trimkv.sh
```

### MMDU-only fine-tune

Smaller mixture (`m4_instruct50_images,mmdu_45k`) tuned for multi-image dialog. Defaults to DBTrimKV on `Qwen/Qwen3-VL-4B-Instruct`:

```bash
GPUS=4 bash scripts/train_trimkv_mmdu.sh
```

## Key training arguments

A few options that aren't obvious from the HF `TrainingArguments` defaults (full list in [argument.py](argument.py)):

- `--retention_gate` — gate variant. `rg` for per-head local-budget TrimKV; `rg10` for global-capacity DBTrimKV.
- `--global_capacity` — `True` enforces `M_global = memory_size × num_layers × num_heads` (DBTrimKV); `False` enforces a local budget per layer/head (TrimKV). Always pair with the matching `retention_gate`.
- `--tie_retention_gate_layers` — `True` ties the gate's final projection across layers/heads (DBTrimKV); `False` keeps per-layer parameters (TrimKV).
- `--memory_size` — KV budget M. Below 1 it is treated as a fraction of the sequence length.
- `--base_loss` — combination of `ntp`, `fwkl`, `rvkl` joined with `_` (e.g. `fwkl_ntp`). KL variants run a vanilla forward pass each step to produce teacher logits.
- `--trainable_params` — pipe-separated suffixes of parameter names to unfreeze (default `self_attn.retention_gate`).
- `--data_flatten` / `--data_packing` — flatten multi-turn conversations and pack short samples up to `model_max_length`. Both default to `True` in the shipped scripts.
- `--logit_block_size` — chunk size for the final logit projection. Use `-1` to disable chunking.
- `--download_from` — where to load TrimKV/DBTrimKV gate weights from when `load_trimkv_weights=True` (`wandb`, `huggingface`, `local`, `none`).

Outputs land in `${OUTPUT_DIR}/models/<run_name>/`, where `<run_name>` encodes the full hyperparameter set.

## Acknowledgements

We thank the following open-source projects for their datasets, code, and resources:

- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL)
- [LLaVA](https://github.com/haotian-liu/LLaVA), [LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K)
- [M4-Instruct-Data](https://huggingface.co/datasets/lmms-lab/M4-Instruct-Data)
- [R1-Onevision](https://huggingface.co/datasets/Fancy-MLLM/R1-Onevision), [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)
- [MMDU](https://huggingface.co/datasets/laolao77/MMDU)
