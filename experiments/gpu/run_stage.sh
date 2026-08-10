#!/usr/bin/env bash
# Staged P3 execution on a 4-GPU host.
#
#   ./run_stage.sh cheap   T-6 kappa + P3-2 ablation + P3-3 head pairs (parallel)
#   ./run_stage.sh seeds   P3-1 re-baseline: seeds 1,2,3 x 4 cells
#
# `cheap` runs first because it is prefill-only and finishes in minutes; the
# seed re-baseline then gets all four cards to itself.
#
# Why seed 1 is re-run here rather than taken from the released logs: the
# reproduction gate showed this host's generations differ from the A100 host's
# on ~8% of inputs (gate decisions are identical; only the decoded text moves).
# Comparing seeds across hosts would report that hardware offset as seed
# variance, so the whole seed family is measured on one host.
set -uo pipefail

ROOT=$HOME/Work/PAGE
SRC=$ROOT/page-kv/experiments/scripts
NEW=$ROOT/page-kv/experiments
OUT=$NEW/results
LOG=$ROOT/page-kv/experiments/logs
mkdir -p "$OUT" "$LOG"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate page-repro

TASKS=niah_multikey_3,vt,fwe,qa_1
BUDGETS=0.0625,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0

STAGE="${1:-}"
case "$STAGE" in

cheap)
  echo "=== stage: prefill-only (T-6, P3-2, P3-3) ==="

  # gpu0 - T-6: margin-to-noise ratio on the capacity-bound class
  CUDA_VISIBLE_DEVICES=0 python "$NEW/scripts/measure_kappa.py" \
    --model Qwen/Qwen2.5-1.5B-Instruct --config 4096 \
    --task niah_multikey_3 --n 20 \
    --out "$OUT/kappa_qwen15b_4k.jsonl" > "$LOG/t6_kappa.log" 2>&1 &

  # gpu1,2 - P3-2: held-out gate-signal ablation
  CUDA_VISIBLE_DEVICES=1 python "$SRC/gate_signal_ablation.py" \
    --model Qwen/Qwen2.5-14B-Instruct --config 4096 --n_per_task 50 --gpu 0 \
    --out "$OUT/signal_ablation_qwen14b_4k.jsonl" > "$LOG/p32_qwen14b.log" 2>&1 &
  CUDA_VISIBLE_DEVICES=2 python "$SRC/gate_signal_ablation.py" \
    --model meta-llama/Llama-3.1-8B-Instruct --config 4096 --n_per_task 50 --gpu 0 \
    --out "$OUT/signal_ablation_llama31_4k.jsonl" > "$LOG/p32_llama31.log" 2>&1 &

  # gpu3 - P3-3: per-head-pair Jaccards
  CUDA_VISIBLE_DEVICES=3 python "$NEW/scripts/head_pair_probe.py" \
    --model Qwen/Qwen2.5-1.5B-Instruct --config 4096 --n_per_task 30 \
    --out "$OUT/head_pairs_qwen15b_4k.jsonl" > "$LOG/p33_pairs.log" 2>&1 &
  wait
  echo "=== cheap stage done ==="

  # Mistral 16K is the OOM risk (48 GiB cards; the paper notes 16K two-pass
  # OOM'd on a contended 80 GiB A100). Try one card, fall back to sharding.
  echo "=== P3-2 third cell: Mistral-7B 16K ==="
  CUDA_VISIBLE_DEVICES=0 python "$SRC/gate_signal_ablation.py" \
    --model mistralai/Mistral-7B-Instruct-v0.3 --config 16384 --n_per_task 30 --gpu 0 \
    --out "$OUT/signal_ablation_mistral7b_16k.jsonl" > "$LOG/p32_mistral16k.log" 2>&1
  if grep -qi "out of memory" "$LOG/p32_mistral16k.log"; then
    echo ">> single-card OOM; retrying sharded across all 4 GPUs"
    CUDA_VISIBLE_DEVICES=0,1,2,3 python "$SRC/gate_signal_ablation.py" \
      --model mistralai/Mistral-7B-Instruct-v0.3 --config 16384 --n_per_task 30 --gpu 0 \
      --out "$OUT/signal_ablation_mistral7b_16k.jsonl" \
      > "$LOG/p32_mistral16k_sharded.log" 2>&1
  fi
  ;;

seeds)
  echo "=== stage: P3-1 seed re-baseline (seeds 1,2,3) ==="
  for seed in 1 2 3; do
    i=0
    for spec in "qwen15b:Qwen/Qwen2.5-1.5B-Instruct" \
                "qwen3b:Qwen/Qwen2.5-3B-Instruct" \
                "mistral7b:mistralai/Mistral-7B-Instruct-v0.3" \
                "qwen14b:Qwen/Qwen2.5-14B-Instruct"; do
      tag=${spec%%:*}; model=${spec#*:}
      echo ">> [gpu$i] seed$seed $tag"
      SHUFFLE_SEED=$seed CUDA_VISIBLE_DEVICES=$i PAGE_SRC="$SRC" \
        python "$NEW/scripts/gated_eviction_seed.py" \
          --model "$model" --config 4096 --tasks "$TASKS" --max_examples 100 \
          --budgets "$BUDGETS" --tau 0.07 --score_policy snapkv --gpu 0 \
          --out "$OUT/rebase_4k_${tag}_seed${seed}.jsonl" \
          > "$LOG/seed${seed}_${tag}.log" 2>&1 &
      i=$((i+1))
    done
    wait
    echo "=== seed $seed complete ==="
  done
  ;;
  *) echo "usage: $0 {cheap|seeds}" >&2; exit 2 ;;
esac
echo "STAGE-DONE:$STAGE"
