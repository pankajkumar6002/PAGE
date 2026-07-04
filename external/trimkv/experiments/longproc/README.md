# LongProc evaluation

This directory evaluates **TrimKV** and a set of decode-time KV-compression baselines (R-KV, SnapKV, StreamingLLM, H2O) on [LongProc](https://github.com/princeton-nlp/LongProc) — Princeton NLP's long-procedural-generation benchmark covering five tasks (countdown search, HTML→TSV extraction, pseudo-code→C++, theory-of-mind tracking, travel planning) at three input-length buckets (0.5k / 2k / 8k).

> **DBTrimKV is not yet supported here.** Only fixed-budget TrimKV is wired up.

The evaluation pipeline is two-stage: `run_longproc.py` generates per-task JSONL outputs, then `eval_and_summarize_results.py` walks the results folder and reports per-dataset metrics using the task-specific evaluators in [longproc/](longproc/).

## Layout

- [run_longproc.py](run_longproc.py) — generation entry point. Loads the chosen method via `load_model.py::LOADER_MAP`, runs the task-specific decoding loop, and writes `results/<dataset>/<run_name>.jsonl`.
- [eval_and_summarize_results.py](eval_and_summarize_results.py) — scoring entry point. Walks a folder of `.jsonl` files and writes a `summary.txt` per dataset using the right `evaluate_*` function from [longproc/](longproc/).
- [load_model.py](load_model.py) — `LOADER_MAP` registry: `fullkv`, `trimkv`, `rkv`, `snapkv`, `streamingllm`, `h2o`, `seerattn`.
- [decoding.py](decoding.py) — generation helpers shared between the methods.
- [longproc/](longproc/) — vendored Python sub-package from [princeton-nlp/LongProc](https://github.com/princeton-nlp/LongProc). Contains the per-task evaluators (`countdown_evaluator.py`, `html_to_tsv_evaluator.py`, `spoc_evaluator.py`, `tom_tracking_evaluator.py`, `travel_planning_evaluator.py`) and the `load_longproc_data` loader.
- [rkv/](rkv/) — vendored R-KV / SnapKV / StreamingLLM / H2O / AnalysisKV compression policies and the corresponding `transformers` monkey-patches.
- [scripts/](scripts/) — `run_trimkv.sh` (TrimKV sweep across the 13 dataset variants), `run_baseline.sh` (single-method one-off).
- [example_usage.py](example_usage.py) — upstream reference example using vLLM + OpenAI; not used by our pipeline but kept for reference.
- `data/` — populated on first `load_longproc_data` call. Gitignored.
- `spoctmp/` — temp `.cpp`/`.bin` files emitted by the spoc evaluator while compiling/running candidate C++ programs. Gitignored.

## Setup

From the repo root:

```bash
pip install -e .
pip install -r experiments/longproc/requirements.txt
```

The pseudo-code → C++ task (`pseudo_to_code_*`) compiles candidate programs through `g++` against the LongProc test cases, so the `spoc_evaluator` requires a working `g++` and writes intermediates to `spoctmp/`.

LongProc data is fetched on-the-fly by `longproc.longproc_data.load_longproc_data` into `./data/` on first run.

## Running

### TrimKV — full 13-dataset sweep

```bash
MODEL=ngocbh/TrimKV-Qwen3-4B-Math \
DOWNFROM=huggingface \
KV_BUDGET=2048 \
N_SAMPLES=1 \
bash scripts/run_trimkv.sh
```

The shipped sweep covers:

```
countdown_{0.5k, 2k, 8k}
pseudo_to_code_{0.5k, 2k}        # spoc compile-and-run, no 8k variant
html_to_tsv_{0.5k, 2k, 8k}
tom_tracking_{0.5k, 2k, 8k}
travel_planning_{2k, 8k}         # no 0.5k variant
```

`DOWNFROM` accepts `huggingface`, `wandb`, or `local`.

### Baseline — single (method, dataset) pair

```bash
MODEL=Qwen/Qwen3-4B-Instruct-2507 \
METHOD=fullkv \
DATANAME=countdown_0.5k \
KV_BUDGET=1024 \
bash scripts/run_baseline.sh
```

Swap `METHOD` to any of `fullkv | rkv | snapkv | streamingllm | h2o | seerattn`. Loop over datasets / methods externally.

### Direct invocation

```bash
python run_longproc.py \
    --dataset countdown_2k \
    --model_path ngocbh/TrimKV-Qwen3-4B-Math \
    --method trimkv \
    --download_from huggingface \
    --kv_budget 2048 \
    --gen_length 32768 \
    --do_sample False \
    --n_samples 1
```

Useful flags: `--resume` (skip already-generated samples; default `True`), `--max_return_sequences N`, `--temperature`/`--top_p`, `--max_model_len`.

### Slurm

`LAUNCHER=slurm` (or `slurm_qos`) submits each per-dataset run via `sbatch scripts/wrapper_resub*.sh python …`. The wrapper scripts are local-only (gitignored); supply your own that match your cluster.

## Scoring

Once the generation JSONLs land under `results/<dataset>/`, score them with:

```bash
python eval_and_summarize_results.py results/
```

The script auto-dispatches to the right evaluator based on the dataset name (countdown, html→tsv, spoc, tom-tracking, travel-planning) and writes a `summary.txt` next to each scored file.

## Acknowledgements

- [LongProc](https://github.com/princeton-nlp/LongProc) (Tan et al.) — the benchmark, the per-task evaluators in [longproc/](longproc/), and the original `example_usage.py` reference.
- [R-KV](https://github.com/Zefan-Cai/R-KV) — the SnapKV / R-KV / StreamingLLM / H2O policies vendored under [rkv/](rkv/).
