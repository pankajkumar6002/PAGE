#!/usr/bin/env bash
# Mistral-7B seed run sharded across every visible GPU on this host.
set -uo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate page-repro
export PAGE_SRC="$HOME/Work/PAGE/page-kv/experiments/scripts"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3}"
R="$HOME/Work/PAGE/page-kv"; cd "$R/experiments"
SHUFFLE_SEED="${SEED:-3}" python scripts/gated_eviction_seed_sharded.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 --config 4096 \
  --tasks niah_multikey_3,vt,fwe,qa_1 --max_examples "${N:-100}" \
  --budgets 0.0625,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0 \
  --tau 0.07 --score_policy snapkv --gpu 0 \
  --out "$R/experiments/results/rebase_4k_mistral7b_seed${SEED:-3}_hostB.jsonl"
echo "SHARDED-EXIT=$?"
