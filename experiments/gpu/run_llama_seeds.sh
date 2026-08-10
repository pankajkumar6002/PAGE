#!/usr/bin/env bash
# Llama-3.1-8B seed replicates, the cross-architecture cell.
#
# Worth having because P3-2 found the task ordering *inverts* on this
# architecture (MK3 mean D above the nearest dilution-prone task). Seed spread
# tells us whether that inversion is stable or an artifact of one input draw.
#
# Seeds 1 and 2 run in parallel on the two free cards; seed 3 follows on
# whichever frees first, so all three share a host and are comparable.
set -uo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate page-repro
export PAGE_SRC="$HOME/Work/PAGE/page-kv/experiments/scripts"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R="$HOME/Work/PAGE/page-kv"; cd "$R/experiments"

run () {  # run <seed> <gpu>
  SHUFFLE_SEED=$1 CUDA_VISIBLE_DEVICES=$2 python scripts/gated_eviction_seed.py \
    --model meta-llama/Llama-3.1-8B-Instruct --config 4096 \
    --tasks niah_multikey_3,vt,fwe,qa_1 --max_examples 100 \
    --budgets 0.0625,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0 \
    --tau 0.07 --score_policy snapkv --gpu 0 \
    --out "$R/experiments/results/rebase_4k_llama31_seed$1.jsonl" \
    > "$R/logs/seed$1_llama31.log" 2>&1
}

run 1 1 & run 2 3 & wait
echo "seeds 1,2 done"
run 3 1
echo "LLAMA-SEEDS-DONE"
