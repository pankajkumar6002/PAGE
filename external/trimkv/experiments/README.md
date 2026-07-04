# Experiments

Per-benchmark evaluation harnesses for TrimKV / DBTrimKV against KV-compression baselines (R-KV, SnapKV, StreamingLLM, H2O, AdaKV, AdaPyramidKV, KeyDiff, Locret, SeerAttention). Each subdirectory is self-contained and ships its own README, run scripts, and vendored baseline code.

## Index

| Benchmark | Variants | Notes |
|---|---|---|
| [math/](math/) | TrimKV, DBTrimKV, R-KV/SnapKV/StreamingLLM/H2O/KeyDiff, SeerAttention | AIME-24, MATH-500, GSM8K. Two-stage: generate → grade with vendored [latex2sympy2](math/evaluation/latex2sympy/). |
| [longbench/](longbench/) | TrimKV, FullKV, Locret + R-KV/SnapKV/StreamingLLM/H2O/KeyDiff | 16-task LongBench. Chunked prefill following Locret. |
| [longbench_v2/](longbench_v2/) | TrimKV, FullKV, Locret + R-KV/SnapKV/StreamingLLM/H2O/KeyDiff | LongBench v2. Same shape as LongBench; harder distribution. |
| [longmemeval/](longmemeval/) | TrimKV + R-KV/SnapKV/StreamingLLM/H2O | Long-term-memory recall. See the directory's "Known caveats" — uses a fixed `DynamicCache`. |
| [longproc/](longproc/) | TrimKV + R-KV/SnapKV/StreamingLLM/H2O | Princeton-NLP LongProc — 5 long-procedural-generation tasks × 3 length buckets (0.5k/2k/8k). DBTrimKV not yet supported. |
| [scbench/](scbench/) | TrimKV + R-KV/SnapKV/StreamingLLM/H2O | Microsoft SCBench — multi-turn / shared-context-different-query (`scdq`) long-context. DBTrimKV not yet supported. |
| [lmms-eval/](lmms-eval/) | TrimKV, DBTrimKV + R-KV/SnapKV/AdaKV/AdaPyramidKV | Fork of [EvolvingLMMs-Lab/lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) — multimodal tasks (mathvision_testmini, video_mmmu_*, mmmu_pro_vision, videomme, videomathqa_mcq, mmstar). |
| [mmdu/](mmdu/) | TrimKV, DBTrimKV + R-KV/SnapKV/AdaKV/AdaPyramidKV | MMDU multi-turn multi-image dialog. Two-stage: generate → judge-LLM score with the rubric in `meta_prompt.txt`. |
| [benchmarks/](benchmarks/) | TrimKV, DBTrimKV + SnapKV/AdaKV/AdaPyramidKV | Decode-time throughput / latency microbenchmark for LLM and VLM Qwen3. Not a quality benchmark — measures `tokens/s` under varying KV budgets and context lengths. |

## Conventions

Most directories follow the same shape:

- **Entry point** — `run_<benchmark>.py` (or `run_benchmark.py`/`run_longmemeval.py`/`run_chunked_prefill.py`/`run_scbench.py`).
- **Loader registry** — `load_model.py::LOADER_MAP` dispatches `--method <name>` to a model-construction function.
- **Vendored baselines** — typically under `rkv/` (R-KV-derived) and/or `baselines/` (CDPruner / PACT / Locret subtrees). Each lives in its own per-experiment copy because the upstream baselines use absolute imports that don't tolerate aliasing.
- **Run scripts** — `scripts/run_trimkv.sh`, `scripts/run_dbtrimkv.sh`, `scripts/run_baselines.sh` etc. All accept `LAUNCHER=python|slurm|...` and override-by-env-var (`MODEL=`, `KV_BUDGET=`, `METHODS=`, `DATANAME=`, …).
- **Slurm wrappers** — `scripts/wrapper*.sh` are gitignored. Each experiment expects users to bring their own.
- **Outputs** — `results/`, `logs/`, `artifacts/` are gitignored.

To reproduce a benchmark, start with that subdirectory's README and the matching `scripts/run_*.sh`.

## Public model checkpoints

The TrimKV / DBTrimKV checkpoints used across these experiments are listed in the [project model registry](../README.md#released-models). All are on Hugging Face and load with `download_from=huggingface`.
