#!/usr/bin/env bash
# Mistral-7B seed 3, the one cell missing from the seed-variance re-baseline.
# HOSTTAG distinguishes the output so the HOST_B and HOST_A runs do not
# overwrite each other: seeds 1 and 2 were measured on HOST_A, so only the
# HOST_A copy is comparable with them.
set -uo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate page-repro
export PAGE_SRC="$HOME/Work/PAGE/page-kv/experiments/scripts"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R="$HOME/Work/PAGE/page-kv"; cd "$R/experiments"
SHUFFLE_SEED=3 CUDA_VISIBLE_DEVICES="${GPU:-1}" python scripts/gated_eviction_seed.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 --config 4096 \
  --tasks niah_multikey_3,vt,fwe,qa_1 --max_examples 100 \
  --budgets 0.0625,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0 \
  --tau 0.07 --score_policy snapkv --gpu 0 \
  --out "$R/experiments/results/rebase_4k_mistral7b_seed3${HOSTTAG:-}.jsonl"
echo "M3-EXIT=$?"
