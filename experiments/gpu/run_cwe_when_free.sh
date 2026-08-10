#!/usr/bin/env bash
# E9 Mistral-7B `cwe` only. Waits for the 3-task Mistral cell to release GPU 0,
# then runs. cwe needs ~61.4 GB; three attempts on a card with 60.98 GB free
# failed by ~0.4 GB, so the wait is not optional.
set -uo pipefail
R2=/home/pankaj/Work/PAGE/page-kv
export PAGE_SRC=/home/pankaj/Work/PAGE/page-kv/experiments/scripts
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate page-repro

echo "waiting for the 3-task Mistral cell to finish..."
while pgrep -f "slug capbound_mistral7b " >/dev/null 2>&1; do sleep 60; done

# Require 62 GB free on GPU 0 before starting.
for i in $(seq 1 240); do
  free=$(nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits -i 0 | awk '{print ($1-$2)/1024}')
  ok=$(echo "$free >= 62" | bc -l)
  [ "$ok" = "1" ] && { echo "GPU 0 has ${free} GB free, starting"; break; }
  echo "  GPU 0 only ${free} GB free, waiting..."; sleep 60
done

CUDA_VISIBLE_DEVICES=0 python "$R2/experiments/scripts/adakv_matrix.py" \
  --model mistralai/Mistral-7B-Instruct-v0.3 --slug capbound_mistral7b_cwe \
  --config 4096 --tasks cwe --max_examples 100 \
  --budgets 0.0625,0.125,0.25,0.375,0.5,0.625,0.75,0.875 \
  --tau 0.07 --floor_alpha 0.5 \
  --out "$R2/experiments/results/capbound_4k_mistral7b_cwe.jsonl" \
  > "$R2/experiments/logs/capbound_mistral7b_cwe.log" 2>&1
echo "cwe exit=$? rows=$(wc -l < "$R2/experiments/results/capbound_4k_mistral7b_cwe.jsonl" 2>/dev/null || echo 0)"
