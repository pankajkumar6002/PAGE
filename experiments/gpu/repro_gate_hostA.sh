#!/usr/bin/env bash
# Step 0: does an A100 host reproduce the released logs?
# The gate was only ever tested on sateri (RTX 6000 Ada), where generations
# drifted on ~8% of inputs. jagannath is an A100, the same class as the machine
# that produced the released logs, so this separates "different architecture"
# from "different machine".
set -uo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate page-repro
R="$HOME/Work/PAGE"; OUT="$R/page-kv/experiments/results"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
python "$R/page-kv/experiments/scripts/gated_eviction.py" \
  --model Qwen/Qwen2.5-1.5B-Instruct --config 4096 \
  --tasks niah_multikey_3,vt,fwe,qa_1 --max_examples 100 \
  --budgets 1.0,0.25 --tau 0.07 --score_policy snapkv --gpu 0 \
  --out "$OUT/repro_hostA.jsonl" 2>&1 | tail -3
python "$R/page-kv/experiments/gpu/repro_diff.py" \
  "$OUT/repro_hostA.jsonl" \
  "$R/page-kv/experiments/results/gated_4k_qwen15b.jsonl"
echo "JG-GATE-EXIT=$?"
