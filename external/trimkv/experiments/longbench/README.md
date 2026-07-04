# LongBench

LongBench evaluation for TrimKV and the Locret / R-KV / SnapKV / StreamingLLM / H2O / KeyDiff baselines, on 16 long-context tasks from [THUDM/LongBench](https://huggingface.co/datasets/THUDM/LongBench).

> **We follow [Locret](https://arxiv.org/abs/2410.01805) for chunked prefilling** — the input prompt is split into fixed-size chunks (default 3072 tokens) and each chunk is consumed sequentially while the KV-eviction policy decides which keys/values to retain at the end of every chunk. This lets methods like Locret / TrimKV evaluate at full LongBench context lengths (up to 128k) with a small KV budget. The chunked-prefill driver lives in [`run_chunked_prefill.py::chunk_prefill_and_generate`](run_chunked_prefill.py); the Locret-specific variant that also re-scores tokens with the retainment head is `locret_chunk_prefill_and_generate` in the same file.

## Layout

- [run_chunked_prefill.py](run_chunked_prefill.py) — entry point. Iterates a single LongBench dataset, runs chunked prefill with the chosen `--method`, and writes per-sample predictions + scores to `results/<run_name>/<dataset>.jsonl`.
- [load_model.py](load_model.py) — `LOADER_MAP` registry: `fullkv`, `trimkv`, `rkv`, `snapkv`, `streamingllm`, `h2o`, `seerattn`, `locret`.
- [metrics.py](metrics.py) — LongBench metric implementations (F1 QA, ROUGE, classification, retrieval, count, code-similarity).
- [gather_results.py](gather_results.py) — collates the per-dataset JSONs under `results/<run_name>/` into a single `all_results.csv`.
- [configs/](configs/) — LongBench config files mirrored from the upstream LongBench repo:
  - `dataset2prompt.json` — task-specific prompt templates.
  - `dataset2maxlen.json` — generation length per task.
  - `model2maxlen.json` — supported context length per base model.
  - `model2path.json` — HF id for each base model.
- [baselines/](baselines/) — vendored baseline code: [`baselines/locret/`](baselines/locret/) (modeling + retainment-head inference) and [`baselines/rkv/`](baselines/rkv/) (R-KV / SnapKV / StreamingLLM / H2O / KeyDiff / AnalysisKV compression policies + monkey-patches).
- [scripts/](scripts/) — three reference launchers (`run_fullkv.sh`, `run_trimkv.sh`, `run_locret.sh`).
- `ckpts/` — Locret retainment-head weights are downloaded here on first run; gitignored, not part of the release.

## Setup

From the repo root:

```bash
pip install -e .
pip install fire datasets rouge-score jieba
```

LongBench datasets are pulled from HF on first use; set `HF_TOKEN` if you hit rate limits.

## Running

Each script iterates a fixed list of 16 LongBench datasets. They are thin wrappers around `run_chunked_prefill.py`; every parameter is overridable via env vars.

### TrimKV

```bash
MODEL=ngocbh/TrimKV-Phi-3-mini-128k-instruct \
DOWNFROM=huggingface \
KV_BUDGET=6000 \
BUFFER_SIZE=0 \
N_SAMPLES=-1 \
bash scripts/run_trimkv.sh
```

`DOWNFROM` accepts `huggingface`, `wandb`, or `local` (matches `--download_from` in `run_chunked_prefill.py`).

### Full-KV baseline

```bash
MODEL=Qwen/Qwen3-4B-Instruct-2507 \
KV_BUDGET=6000 \
N_SAMPLES=-1 \
bash scripts/run_fullkv.sh
```

`KV_BUDGET` is ignored when `--method fullkv` (no eviction); the default value just keeps the run-name format consistent.

### Locret

```bash
MODEL=hyx21/Locret-phi-3-mini-128K \
KV_BUDGET=6000 \
STABILIZERS=2500 \
N_SAMPLES=-1 \
bash scripts/run_locret.sh
```

The script downloads the Phi-3-mini-128k retainment-head weights (`phi-3-mini-128K.bin`) from `hyx21/Locret-phi-3-mini-128K` into `ckpts/locret/<model_type>/` on first run.

### Slurm

Set `LAUNCHER=slurm` to submit each per-dataset run via `sbatch scripts/wrapper_resub.sh python …`. The `wrapper.sh`, `wrapper_resub.sh`, and `wrapper_cpu.sh` wrappers are local-only (gitignored); add your own that match your cluster.

## Direct invocation

For ad-hoc runs, call `run_chunked_prefill.py` directly:

```bash
python run_chunked_prefill.py \
    --dataset hotpotqa \
    --model_type phi3-mini-128k \
    --model_path ngocbh/TrimKV-Phi-3-mini-128k-instruct \
    --method trimkv \
    --download_from huggingface \
    --kv_budget 6000 \
    --buffer_size 0 \
    --chunk_size 3072 \
    --n_samples -1
```

`--chunk_size` is the chunked-prefill window from Locret (default 3072 tokens). `--n_samples=-1` runs the full dataset; pass a positive integer for a quick smoke test.

## Collating results

After all 16 datasets finish, gather the JSONs into one CSV:

```bash
python gather_results.py --results_dir results/
```

The script parses run names of the form `<method>-<chunk_size>-<budget>b[-<buffer>bf]-<max_seq_len>l-<seed>` and writes `all_results.csv` with one row per (run, dataset, metric).

## Acknowledgements

- [LongBench](https://github.com/THUDM/LongBench) — the benchmark and the prompt/metric configs in [configs/](configs/).
- [Locret](https://github.com/huangyuxiang03/Locret) — the chunked-prefill methodology and the retainment-head baseline; vendored under [baselines/locret/](baselines/locret/).
- [R-KV](https://github.com/Zefan-Cai/R-KV) — the R-KV / SnapKV / StreamingLLM / H2O baseline implementations vendored under [baselines/rkv/](baselines/rkv/).
