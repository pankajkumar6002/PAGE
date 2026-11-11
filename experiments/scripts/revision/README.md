# Revision experiments and checks

This folder reproduces the results added in the revision: the re-checks of published values (Tables 1 and 4,
Figure 4), the analyses of the released logs (Tables 33 to 40, 42, 46 and 49), and the new GPU runs (Tables 41,
43 to 45, 47 and 48). Every path below is relative to the repository root, and every command is run from there.

## Layout

| Path | Contents |
|---|---|
| `experiments/scripts/revision/checks/` | Log-only checks of the released logs in `experiments/results/` |
| `experiments/scripts/revision/` | GPU runners and the analyses of the new runs |
| `experiments/results/revision/` | Logs of the new runs (`*.jsonl`), analysis outputs (`*.md`), expected check outputs (`checks/*.out`) and runner logs (`logs/*.log`) |
| `experiments/data/revision/` | `fetch.sh` for the third-party data, its checksums and licence notices |
| `preregistration/revision/` | Pre-registrations of B1, B2, B2b, B2c, B3, B7 and B8, with their recorded SHA-256 |

## CPU-only reproduction

These commands need only `numpy`. They read logs that are already in the repository and write nothing except
standard output.

```bash
python3 -I experiments/scripts/revision/checks/verify_2026-11-07.py   > /tmp/verify.out
python3 -I experiments/scripts/revision/checks/table1_check.py        > /tmp/table1.out
python3 -I experiments/scripts/revision/checks/layer_profile_check.py > /tmp/layer_profile.out
python3 -I experiments/scripts/revision/analyze_b.py b1    > /tmp/b1_analysis.md
python3 -I experiments/scripts/revision/analyze_b.py b2    > /tmp/b2_analysis.md
python3 -I experiments/scripts/revision/analyze_b.py b2c   > /tmp/b2c_analysis.md
python3 -I experiments/scripts/revision/analyze_b.py b3    > /tmp/b3_analysis.md
python3 -I experiments/scripts/revision/analyze_b.py b5 b6 > /tmp/b5_b6_analysis.md
python3 -I experiments/scripts/revision/analyze_b7.py      > /tmp/b7_analysis.md
python3 -I experiments/scripts/revision/analyze_b8_corrected.py > /tmp/b8_analysis.md
```

Each output equals the corresponding file in `experiments/results/revision/` (`checks/*.out` for the three
checks), with two exceptions:
- `b8_analysis.md` ends with a table of the pre-registered predictions, B8-T6, scored by hand from B8-T1 to B8-T5.
- One line of `b7_analysis.md` reports how many L-Eval quality articles and questions also appear in the B2
  QuALITY sample. It needs the L-Eval files from `fetch.sh` and `emozilla/quality` in the local Hugging Face
  cache (the script reads it offline). Without them, that line says the overlap was not computed. Every table
  is unaffected. `verify_2026-11-07.py` exits non-zero if any assertion against a
published value fails. `PAGE_RESULTS` overrides the released-log folder for the checks, and
`PAGE_REV_RESULTS` overrides the revision-log folder for `analyze_b.py`.

## Map from the paper to the code

| Paper | Script and command | Table in the output | Logs | Pre-registration |
|---|---|---|---|---|
| Table 1 | `checks/table1_check.py` | per-cell rho, pooled NIAH-MK3 count | released RULER sweep, gated and partition logs | none |
| Table 4 | `checks/verify_2026-11-07.py --sections A0` | A0-T1 | released logs | none |
| Figure 4 | `checks/layer_profile_check.py` | early and late third per task | `drops_*_n100.jsonl`, `drops_llama31_8b_4k.jsonl` | none |
| Table 33 | `checks/verify_2026-11-07.py --sections A1` | A1-T1 and A1-T2 | released 4x4 matrix logs | none |
| Table 34 | `checks/verify_2026-11-07.py --sections A2` | A2-T2 | released 4x4 matrix logs | none |
| Table 35 | same | A2-T1 | same | none |
| Table 36 | same | A2-T3 | released logs of the other cells | none |
| Table 37 | `checks/verify_2026-11-07.py --sections A3` | A3-T1 | released logs | none |
| Table 38 | same | A3-T2 | same | none |
| Table 39 | `checks/verify_2026-11-07.py --sections A4` | A4-T1 | released calibration logs | none |
| Table 40 | same | A4-T2 | same | none |
| Table 41 | `analyze_b.py b1` | B1-T1 | `b1_signals_{qwen15b,qwen3b,mistral7b,qwen14b}.jsonl`, joined with the released `gated_4k_*.jsonl` | `b1_scorer_gates_prereg.md` |
| Table 42 | `checks/verify_2026-11-07.py --sections A7` | A7-T1 | released logs | none |
| Table 43 | `analyze_b.py b3` | B3-T1 | `b3_qwen3_14b.jsonl` | `b3_b4_models_prereg.md` |
| Table 44 | `analyze_b.py b5 b6` | B5-T0, B6-T0, per-model T1 and T4 | `b5_olmo_hybrid_7b.jsonl`, `b5_qwen3_next_80b.jsonl`, `b5_lfm2_24b_a2b.jsonl`, `b6_deepseek_v2_lite.jsonl`, `b6_glm47_flash.jsonl` | none |
| Table 45 | `analyze_b8_corrected.py` | B8-T1 to B8-T3 | `b8_{nq30,babilong}_{qwen3b,mistral7b,qwen14b}.jsonl`, `b7_leval_{qwen3b,mistral7b,qwen14b}.jsonl`, `logs/*.log` | `b8_neartie_prereg.md` |
| Table 46 | `checks/verify_2026-11-07.py --sections A0` | A0-T2 | `capbound_4k_mistral7b_cwe.jsonl` | none |
| Table 47 | `analyze_b.py b2` and `analyze_b.py b2c` | B2-T1 and B2-T3 (letter, quote and LongBench rows), B2c-T1 and B2c-T3 (Qwen2.5-14B QuALITY rows) | `b2_{quality,lbv2}_{qwen3b,mistral7b}.jsonl`, `b2b_{quality,lbv2}_{qwen3b,mistral7b}.jsonl`, `b2_longbench_{qwen3b,qwen14b}.jsonl`, `b2_longbench_qwen14b_rougeL.jsonl`, `b2c_quality_qwen14b_{letter,quote}.jsonl` | `b2_beyond_ruler_prereg.md`, `b2b_reasoning_prompt_prereg.md`, `b2c_quality_qwen14b_prereg.md` |
| Table 48 | `analyze_b7.py` | pooled rows per model and arm (B7-T1); the table note uses B7-T2 | `b7_{leval,quote}_{qwen3b,mistral7b,qwen14b,qwen3_14b,llama31_8b}.jsonl`, `logs/b7_*.log` | `b7_leval_prereg.md` |
| Table 49 | `checks/verify_2026-11-07.py --sections A6` | A6-T1 | released LongBench logs | none |

Table 47 leaves out gov_report, because substring accuracy is zero at full cache. Its rows are in
`b2_longbench_qwen14b.jsonl` and in `b2_analysis.md` (ROUGE-L in B2-T2, from `score_rouge.py`, which needs
`rouge==1.0.1`).

Table 44 columns: AUC(D) for MK3, MK3 closed, tau open and kept at 1/2 are in B5-T0 and B6-T0. Selection at
b = 1/16 is in B5-T0 and B6-T0. Selection averaged over budgets is the z-gate row of each model's T1 table.

Table 45 uses `analyze_b8_corrected.py`. It runs the pre-registered `analyze_b8.py` unchanged and adds one
post-hoc correction to the reported p-values: the two-sided bootstrap p becomes
min(1, 2 (min(n_nonpositive, n_nonnegative) + 1) / (R + 1)) before the Holm adjustment. Estimates and intervals
are unchanged. `python3 -I experiments/scripts/revision/analyze_b8.py` prints the uncorrected values.
`analyze_b8.py` treats a run as complete only if its runner log in `experiments/results/revision/logs/` has
the runner's `done` line.

## GPU runs

Requirements are those of the main README (Python 3.13, `requirements.txt`, CUDA build of torch).
`b8_neartie.py` also uses `regex`, which `transformers` installs. Models load with `trust_remote_code=False`. The runners import the
released `experiments/scripts/gated_eviction.py` and `longbench_gating.py` (override the folder with
`PAGE_SRC`). Each runner appends to `--out` and refuses or resumes as its docstring states.

Data for Tables 45 and 48 is not in the repository. Fetch it first (about 170 MB, checked against the SHA256SUMS
files):

```bash
bash experiments/data/revision/fetch.sh
```

`PAGE_B8_RAW` and `PAGE_LEVAL_DIR` override the data folders.

```bash
R=experiments/results/revision
S=experiments/scripts/revision

# Table 41 (B1). Qwen2.5-14B with --n_per_task 50.
python $S/b1_scorer_signals.py --gpu 0 --model Qwen/Qwen2.5-1.5B-Instruct --logged experiments/results/gated_4k_qwen15b.jsonl --out $R/b1_signals_qwen15b.jsonl
python $S/b1_scorer_signals.py --gpu 0 --model Qwen/Qwen2.5-3B-Instruct   --logged experiments/results/gated_4k_qwen3b.jsonl  --out $R/b1_signals_qwen3b.jsonl
python $S/b1_scorer_signals.py --gpu 0 --model mistralai/Mistral-7B-Instruct-v0.3 --logged experiments/results/gated_4k_mistral7b.jsonl --out $R/b1_signals_mistral7b.jsonl
python $S/b1_scorer_signals.py --gpu 0 --model Qwen/Qwen2.5-14B-Instruct --n_per_task 50 --logged experiments/results/gated_4k_qwen14b.jsonl --out $R/b1_signals_qwen14b.jsonl

# Table 43 (B3). The released runner with thinking off in the chat template.
python $S/b3_run.py --gpu 0 --model Qwen/Qwen3-14B --tasks niah_multikey_3,vt,fwe,qa_1 --budgets 1.0,0.5,0.25,0.125,0.0625 \
    --max_examples 50 --two_pass --attn_impl sdpa --out $R/b3_qwen3_14b.jsonl

# Table 44 (B5, B6). --expect_kv is the number of layers with a key-value cache.
# Models that need more than one card: --gpu takes a comma list (main card first) with one --mem_gb cap per card.
python $S/b5_arch.py --gpu 0 --model allenai/Olmo-Hybrid-Instruct-DPO-7B --attn eager --expect_kv 8  --out $R/b5_olmo_hybrid_7b.jsonl
python $S/b5_arch.py --gpu 0,1,2,3 --mem_gb <one cap per card> --model Qwen/Qwen3-Next-80B-A3B-Instruct --attn eager --expect_kv 12 --out $R/b5_qwen3_next_80b.jsonl
python $S/b5_arch.py --gpu 0 --model LiquidAI/LFM2-24B-A2B --attn eager --expect_kv 10 --out $R/b5_lfm2_24b_a2b.jsonl
python $S/b5_arch.py --gpu 0 --model deepseek-ai/DeepSeek-V2-Lite-Chat --attn eager --expect_kv 27 --out $R/b6_deepseek_v2_lite.jsonl
python $S/b5_arch.py --gpu 0,1 --mem_gb 54,12 --model zai-org/GLM-4.7-Flash --attn eager --expect_kv 47 --check_b_tol 4.0 --out $R/b6_glm47_flash.jsonl
# The float32 CPU check behind GLM-4.7-Flash's check (b) tolerance, and the routing diagnostic,
# for a task T in niah_multikey_3, vt, fwe, qa_1:
python $S/b6_fp32_checkb.py --model zai-org/GLM-4.7-Flash --task T
python $S/b6_glm_ctx_routing.py --model zai-org/GLM-4.7-Flash --task T

# Table 45 (B8). The L-Eval background of each mixed workload, then the near-tie datasets.
for m in Qwen/Qwen2.5-3B-Instruct:qwen3b mistralai/Mistral-7B-Instruct-v0.3:mistral7b Qwen/Qwen2.5-14B-Instruct:qwen14b; do
  id=${m%%:*}; s=${m##*:}
  python $S/b7_leval.py --gpu 0 --attn_impl sdpa --two_pass --arm leval --model $id --out $R/b7_leval_$s.jsonl > $R/logs/b7_leval_$s.log 2>&1
  for d in nq30 babilong; do
    python $S/b8_neartie.py --gpu 0 --two_pass --dataset $d --model $id --out $R/b8_${d}_$s.jsonl > $R/logs/b8_${d}_$s.log 2>&1
  done
done
```

```bash
# Table 47 (B2 letter arm; B2b quote arm with --mcq_style reasoning; B2c on Qwen2.5-14B).
# QuALITY, LongBench-v2 and LongBench are loaded from the Hugging Face Hub at run time
# (emozilla/quality, zai-org/LongBench-v2, Xnhyacinth/LongBench), without a pinned revision.
for m in Qwen/Qwen2.5-3B-Instruct:qwen3b mistralai/Mistral-7B-Instruct-v0.3:mistral7b; do
  id=${m%%:*}; s=${m##*:}
  python $S/b2_gating.py --gpu 0 --attn_impl sdpa --two_pass --dataset quality --n 300 --model $id --out $R/b2_quality_$s.jsonl
  python $S/b2_gating.py --gpu 0 --attn_impl sdpa --two_pass --dataset lbv2 --model $id --out $R/b2_lbv2_$s.jsonl
  python $S/b2_gating.py --gpu 0 --attn_impl sdpa --two_pass --mcq_style reasoning --dataset quality --n 300 --model $id --out $R/b2b_quality_$s.jsonl
  python $S/b2_gating.py --gpu 0 --attn_impl sdpa --two_pass --mcq_style reasoning --dataset lbv2 --model $id --out $R/b2b_lbv2_$s.jsonl
done
python $S/b2_gating.py --gpu 0 --attn_impl sdpa --two_pass --dataset longbench --tasks lcc,repobench-p,hotpotqa --n 100 \
    --model Qwen/Qwen2.5-3B-Instruct --out $R/b2_longbench_qwen3b.jsonl
python $S/b2_gating.py --gpu 0 --attn_impl sdpa --two_pass --dataset longbench --tasks 2wikimqa,musique,gov_report --n 100 \
    --max_context_tokens 24000 --model Qwen/Qwen2.5-14B-Instruct --out $R/b2_longbench_qwen14b.jsonl
python -I $S/score_rouge.py $R/b2_longbench_qwen14b.jsonl $R/b2_longbench_qwen14b_rougeL.jsonl   # needs rouge==1.0.1
python $S/b2_gating.py --gpu 0 --attn_impl sdpa --two_pass --dataset quality --n 300 \
    --model Qwen/Qwen2.5-14B-Instruct --out $R/b2c_quality_qwen14b_letter.jsonl
python $S/b2_gating.py --gpu 0 --attn_impl sdpa --two_pass --mcq_style reasoning --dataset quality --n 300 \
    --model Qwen/Qwen2.5-14B-Instruct --out $R/b2c_quality_qwen14b_quote.jsonl

# Table 48 (B7). Both arms on five models. The leval arm of the first three models is already run for Table 45.
# Qwen3-14B runs with --no_thinking. meta-llama/Llama-3.1-8B-Instruct is gated (see the main README).
for m in Qwen/Qwen2.5-3B-Instruct:qwen3b mistralai/Mistral-7B-Instruct-v0.3:mistral7b Qwen/Qwen2.5-14B-Instruct:qwen14b \
         meta-llama/Llama-3.1-8B-Instruct:llama31_8b Qwen/Qwen3-14B:qwen3_14b; do
  id=${m%%:*}; s=${m##*:}; extra=""; [ $s = qwen3_14b ] && extra="--no_thinking"
  for arm in leval quote; do
    [ -s $R/b7_${arm}_$s.jsonl ] && grep -q '^b7 done' $R/logs/b7_${arm}_$s.log 2>/dev/null && continue
    python $S/b7_leval.py --gpu 0 --attn_impl sdpa --two_pass $extra --arm $arm --model $id --out $R/b7_${arm}_$s.jsonl > $R/logs/b7_${arm}_$s.log 2>&1
  done
done
```

Then rerun the CPU commands above.

## Pre-registrations and their hashes

`preregistration/revision/` holds the pre-registrations of B1 (Table 41), B2 and B2b (Table 47), B2c (the
Qwen2.5-14B QuALITY rows of Table 47), B3 (Table 43), B7 (Table 48, and the L-Eval half of Table 45) and B8
(Table 45). Each was hashed before its runs. Paths inside the `.sha256` files are relative to the original
layout, where `preregistration/` and `experiments/` were siblings: `../experiments/scripts/X` is
`experiments/scripts/revision/X` here, and `../data/Y` is `experiments/data/revision/Y`.

For this anonymous release, some files were edited after hashing. Edits to the pre-registrations replace a
local path, a server-specific file name, an environment name or another group's name, and nothing else:

| File | Edit | Recorded hash |
|---|---|---|
| `b1_scorer_gates_prereg.md` | line 11: a local file path replaced by "internal analysis notes, not released" | no longer matches |
| `b2_beyond_ruler_prereg.md` | line 11: a local file path replaced by "internal analysis notes, not released" | no longer matches |
| `b2b_reasoning_prompt_prereg.md`, `b2c_quality_qwen14b_prereg.md` | none | matches |
| `b3_b4_models_prereg.md` | line 18: a local log path replaced by "log not released" | no longer matches |
| `b7_leval_prereg.md` | line 11: a local file path replaced by "internal analysis notes, not released"; line 31: an environment name replaced by "the same Python environment"; line 108: two launcher file names replaced by "a watchdog script (not released)" and "a queue file (not released)" | no longer matches |
| `b7_leval_prereg.sha256` | lines 5, 9 and 11: file paths rewritten (absolute paths made relative, a server-specific file name removed); every hash value unchanged | n/a |
| `b8_neartie_prereg.md` | line 3: a local folder path replaced by "internal notes, not released"; lines 55 and 119: the name of another group's jobs replaced by "other users' jobs" | no longer matches |
| `b2_gating.py`, `b7_leval.py`, `analyze_b7.py`, `b8_neartie.py`, `analyze_b8.py`, `checks/verify_2026-11-07.py` | absolute paths replaced by paths relative to the repository; docstring paths updated; verify as listed below | no longer match |

No prediction, threshold, analysis rule or code path changed. These hashed files still match:
`experiments/scripts/longbench_gating.py`, `experiments/data/revision/b8_raw/SHA256SUMS` and
`experiments/data/revision/leval/SHA256SUMS`. `b7_leval_prereg.sha256` also lists launcher scripts used for the
runs, which are not released.

## Other edits to released copies

- All scripts: absolute paths replaced by paths relative to the repository, with the existing environment
  overrides kept.
- `checks/verify_2026-11-07.py` prints its log folder relative to the repository.
- `analyze_b.py` and `checks/verify_2026-11-07.py` contain only the sections behind the paper's tables.
  Every section seeds its own random generator, so the remaining sections give the same values.
- `b5_arch.py` contains only the options used for the reported runs.
