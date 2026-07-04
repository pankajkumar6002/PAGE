# train/llm — TrimKV training for text LLMs

This directory trains the TrimKV (and DBTrimKV) retention-gate on top of a frozen text LLM (Qwen3, Qwen2, Llama, Phi-3) using DeepSpeed + 🤗 Trainer. Only the gating parameters are updated; the base model stays frozen, which keeps memory low even at 128k context.

## TrimKV vs DBTrimKV

Both variants share the same training loop, the same datasets, and the same loss surface — they differ only in **how the KV budget is allocated** and **which retention-gate parameterisation is used**. You select between them with two env vars when launching either script:

| | TrimKV | DBTrimKV |
|---|---|---|
| `RETENTION_GATE` | `rg` | `rg10` |
| `GLOBAL_CAPACITY` | `False` | `True` |
| Budget semantics | per-layer, per-head local budget `M_local = MEMORY_SIZE` | single global budget `M_global = MEMORY_SIZE × num_layers × num_heads`, redistributed dynamically across layers/heads |
| Gate parameterisation | independent retention gate per head | final projection of the gate **tied across layers and heads** |
| Typical `MEMORY_SIZE` in the paper | 256 | 128 |

Everything else (`base_model`, `dataset_name`, `training_max_length`, `base_loss`, `lr`, …) is orthogonal — pick whichever recipe fits the data you want to train on, then flip those two flags to choose TrimKV vs DBTrimKV.

## Layout

- [train.py](train.py) — entry point. Defines `TrimKVTrainer`, `ModelArguments`, and `TrainingArguments`. Wraps the base model in the matching `TrimKV*ForCausalLM` class and trains the parameters selected by `--trainable_params`.
- [dataset/](dataset/) — dataset loaders. The `DATASET_LOADER` registry in [dataset/__init__.py](dataset/__init__.py) currently exposes:
  `openr1_math`, `ultrachat`, `synth_long`, `long_alpaca`, `buddhi`, `booksum`, `niah`. Multiple datasets can be concatenated by passing them as a comma-separated list to `--dataset_name`.
- [chat_template/](chat_template/) — overrides for tokenizers whose default chat template strips `<think>…</think>` content. See [chat_template/README.md](chat_template/README.md) for the rationale.
- [ds_config/](ds_config/) — DeepSpeed configs. `stage2.json` is the supported default; `stage3.json` is currently broken with this code path.
- [scripts/](scripts/) — launchers for the two training recipes.

## Setup

Install the top-level `trimkv` package (from the repo root) so the `trimkv.models.*` imports resolve, then create the env file used by the launch scripts:

```bash
# from repo root
pip install -e .

# in this directory
cp ../../.env ./    # provides WANDB_PROJECT, WANDB_API_KEY, WANDB_USERNAME
```

Both training scripts `export $(cat .env | xargs)` at the top, so any extra variables you set there (e.g. `HF_HOME`, `HF_TOKEN`) propagate to the run.

## Training recipes

Both scripts are thin wrappers around `torchrun … train.py …`. Every parameter is overridable via env vars; defaults are set at the top of each script. The script you pick determines the **data + sequence length**; the env vars in the table above determine **TrimKV vs DBTrimKV**.

### Long-context recipe (`scripts/train_trimkv_long.sh`)

KL distillation on long documents (`synth_long`, `booksum`, `buddhi`) at up to 131k tokens. Defaults are tuned for **TrimKV** (`RETENTION_GATE=rg`, `GLOBAL_CAPACITY=False`, `MEMORY_SIZE=256`).

Train **TrimKV** on long context:

```bash
GPUS=4 BS=1 GAS=1 \
DATASET_NAME=synth_long,booksum,buddhi \
TRAINING_MAX_LENGTH=131072 \
MEMORY_SIZE=256 \
RETENTION_GATE=rg \
GLOBAL_CAPACITY=False \
BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507 \
bash scripts/train_trimkv_long.sh
```

Train **DBTrimKV** on the same data — only the gate / capacity flags change:

```bash
GPUS=4 BS=1 GAS=1 \
DATASET_NAME=synth_long,booksum,buddhi \
TRAINING_MAX_LENGTH=131072 \
MEMORY_SIZE=128 \
RETENTION_GATE=rg10 \
GLOBAL_CAPACITY=True \
BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507 \
bash scripts/train_trimkv_long.sh
```

See [scripts/train_trimkv_long.sh](scripts/train_trimkv_long.sh) for the full env-var surface (`RG_BIAS_INIT`, `BASE_LOSS`, `LR`, `LOGIT_BLOCK_SIZE`, …).

### Math-reasoning recipe (`scripts/train_trimkv_math.sh`)

R1-style reasoning traces (`openr1_math`) at 32k tokens. Defaults are tuned for **DBTrimKV** (`RETENTION_GATE=rg10`, `GLOBAL_CAPACITY=True`, `MEMORY_SIZE=128`).

Train **DBTrimKV** on math (default):

```bash
GPUS=4 \
DATASET_NAME=openr1_math \
TRAINING_MAX_LENGTH=32768 \
BASE_MODEL=Qwen/Qwen3-4B \
bash scripts/train_trimkv_math.sh
```

Train **TrimKV** on the same data:

```bash
GPUS=4 \
DATASET_NAME=openr1_math \
TRAINING_MAX_LENGTH=32768 \
MEMORY_SIZE=256 \
RETENTION_GATE=rg \
GLOBAL_CAPACITY=False \
BASE_MODEL=Qwen/Qwen3-4B \
bash scripts/train_trimkv_math.sh
```

See [scripts/train_trimkv_math.sh](scripts/train_trimkv_math.sh) for the full env-var surface.

### Debug runs

`DEBUG=1` shortens any recipe to 10 steps with `max_samples=100`, disables W&B, and lowers `training_max_length` so it fits on a single GPU.

```bash
DEBUG=1 GPUS=1 bash scripts/train_trimkv_long.sh
```

## Key training arguments

A few options that aren't obvious from the HF `TrainingArguments` defaults:

- `--retention_gate` — gate variant. `rg` for per-head local-budget TrimKV; `rg10` for global-capacity DBTrimKV (final projection tied across layers and heads).
- `--global_capacity` — `True` enforces `M_global = memory_size × num_layers × num_heads` (DBTrimKV). `False` enforces a local budget per layer/head (TrimKV). Always pair with the matching `retention_gate`.
- `--memory_size` — KV budget M. Below 1 it is treated as a fraction of the sequence length.
- `--base_loss` — combination of `ntp`, `fwkl`, `rvkl` joined with `_` (e.g. `fwkl_ntp`). KL variants additionally run a vanilla forward pass on each step to produce teacher logits.
- `--trainable_params` — pipe-separated suffixes of parameter names to unfreeze (default `self_attn.retention_gate`).
- `--logit_block_size` — chunk size for the final logit projection. Use `-1` to disable chunking.
- `--data_packing` — set `True` to pack short samples up to `training_max_length`.

Outputs land in `${OUTPUT_DIR:-./models}/<base_name>/<run_name>/`, where `<run_name>` encodes the full hyperparameter set.
