# Math reasoning evaluation

This directory evaluates TrimKV / DBTrimKV and a set of decode-time KV-compression baselines (R-KV, SnapKV, StreamingLLM, H2O, KeyDiff, SeerAttention) on long chain-of-thought reasoning tasks: **AIME-24**, **MATH-500**, and **GSM8K**.

The eval pipeline is two-stage: `run_math.py` generates samples per (method, KV budget) configuration and writes JSONL outputs, then `eval_math.py` scores them with the LongBench-style `latex2sympy2` math-equivalence checker.

## Layout

- [run_math.py](run_math.py) — generation entry point. Loads the configured method via `load_model.py::LOADER_MAP` and writes `results/<run_name>/<dataset>_<seed>.jsonl`.
- [eval_math.py](eval_math.py) — scoring entry point. Reads the JSONL outputs and writes a per-run metrics JSON next to them.
- [load_model.py](load_model.py) — `LOADER_MAP` registry: `fullkv`, `trimkv`, `dbtrimkv`, `rkv`, `snapkv`, `streamingllm`, `h2o`, `keydiff`, `seerattn`.
- [generation_utils.py](generation_utils.py), [utils.py](utils.py) — shared helpers (batched generation, batch-size estimation).
- [data/](data/) — three input JSONL files: `aime24.jsonl`, `math.jsonl` (MATH-500), `gsm8k.jsonl`.
- [evaluation/](evaluation/) — math grading toolkit (parser, grader, executor, data loader). The `evaluation/latex2sympy/` subtree is a vendored copy of [latex2sympy2](https://github.com/OrangeX4/latex2sympy) used for equation equivalence.
- [rkv/](rkv/) — vendored R-KV / SnapKV / StreamingLLM / H2O / KeyDiff compression policies and their monkey-patches over `transformers`.
- [scripts/](scripts/) — `run_trimkv.sh`, `run_dbtrimkv.sh`, `run_baselines.sh`, `run_seerattn.sh`, `eval.sh`.

## Setup

From the repo root:

```bash
pip install -e .
pip install -r experiments/math/requirements.txt
```

Build the math-equivalence checker once:

```bash
cd experiments/math/evaluation/latex2sympy
pip install -e .
cd ..
pip install -r requirements.txt
```

## Running

### TrimKV

```bash
DATANAME=aime24 \
MODEL=ngocbh/TrimKV-Qwen3-4B-Math \
DOWNFROM=huggingface \
N_SAMPLES=8 \
bash scripts/run_trimkv.sh
```

The script sweeps `KV_BUDGET_SET` (256/512/1024/2048/4096 for AIME-24, 64/128/512/1024/2048 for MATH/GSM8K) and writes one `results/<run_name>/...` directory per budget.

### DBTrimKV

```bash
DATANAME=aime24 \
MODEL=ngocbh/DBTrimKV-Qwen3-4B-Math \
DOWNFROM=huggingface \
N_SAMPLES=8 \
bash scripts/run_dbtrimkv.sh
```

### Baselines (FullKV, R-KV, SnapKV, StreamingLLM, KeyDiff)

```bash
DATANAME=aime24 \
MODEL=Qwen/Qwen3-4B \
N_SAMPLES=64 \
bash scripts/run_baselines.sh
```

Iterates `METHOD_SET=(fullkv keydiff rkv snapkv streamingllm)` × the KV-budget sweep. FullKV exits the inner loop after one run since it's budget-independent.

### SeerAttention

```bash
DATANAME=aime24 \
MODEL=SeerAttention/SeerAttention-Decode-Qwen3-4B-AttnGates \
bash scripts/run_seerattn.sh
```

### Direct invocation

```bash
python run_math.py \
    --dataset aime24 \
    --model_path ngocbh/TrimKV-Qwen3-4B-Math \
    --method trimkv \
    --download_from huggingface \
    --kv_budget 1024 \
    --attn_implementation flash_attention_2 \
    --n_samples 8
```

### Slurm

`LAUNCHER=slurm` (or `slurm_nmi` in `run_dbtrimkv.sh`) submits each per-budget run via `sbatch scripts/wrapper_resub*.sh python …`. The wrapper scripts are local-only (gitignored); supply your own that match your cluster.

## Scoring

Once a run directory contains generation JSONLs, score them with:

```bash
DATANAME=aime24 CPUS=4 bash scripts/eval.sh path/to/results_dir/
```

This wraps `eval_math.py`, which uses [`evaluation/grader.py`](evaluation/grader.py) + the latex2sympy checker to mark each sample correct/incorrect and writes a `_cot_metrics.json` summary per run.

## Known caveats

We observed a bug in the upstream R-KV `DynamicCache` implementation: `past_key_values.get_seq_length()` returns the length of the **currently cached** tokens rather than the total tokens seen so far, which produces wrong `cache_position` values during decoding and degrades attention. We use a fixed version of `DynamicCache` for the experiments reported in our paper.

The math results in our paper were collected with the **original** R-KV `DynamicCache` (i.e. before the fix). If you re-run these experiments with the patched cache, expect small numerical differences from the paper. We do not currently have bandwidth to rerun all baseline configurations; treat the R-KV / SnapKV combinations here as best-effort.

## Acknowledgements

- Generation/grading scaffolding adapted from [R-KV](https://github.com/Zefan-Cai/R-KV) (Cai et al., arXiv 2505.24133).
- [latex2sympy2](https://github.com/OrangeX4/latex2sympy) (vendored) for math expression equivalence.
- AIME-24, MATH, GSM8K — the standard math reasoning benchmarks.
