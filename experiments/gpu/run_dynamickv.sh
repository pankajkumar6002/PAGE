#!/usr/bin/env bash
# P3-4 DynamicKV head-to-head. Kept as a file rather than an inline tmux
# command because nested quoting through ssh -> bash -lc -> tmux has silently
# swallowed the command twice now.
set -uo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate page-repro
export PAGE_SRC="$HOME/Work/PAGE/page-kv/experiments/scripts"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R="$HOME/Work/PAGE/page-kv"
cd "$R/experiments"
CUDA_VISIBLE_DEVICES="${GPU:-0}" python scripts/dynamickv_headtohead.py \
  --model "${MODEL:-Qwen/Qwen3-4B-Instruct-2507}" --config 4096 \
  --max_examples "${N:-50}" --tau 0.07 --gpu 0 \
  --out "$R/experiments/results/dynamickv_qwen3_4k.jsonl"
echo "DKV-EXIT=$?"
