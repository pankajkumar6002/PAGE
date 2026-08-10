#!/usr/bin/env bash
# P3 GPU runs, sharded across the 4 idle cards on this host.
#
#   ./run_p3.sh gate     reproduction gate  (must PASS before seeds)
#   ./run_p3.sh cheap    T-6 kappa, P3-2 signal ablation, P3-3 head pairs
#   ./run_p3.sh seeds    P3-1 seed replicates
#   ./run_p3.sh all      gate, then cheap, then seeds
#
# Each cell is pinned to one GPU and run in parallel; 48 GiB per card is
# ample for everything except possibly Mistral-7B at 16K, which is flagged.
set -uo pipefail

ROOT=$HOME/Work/PAGE
SRC=$ROOT/page-kv/experiments
OUT=$ROOT/page-kv/experiments/results
LOG=$ROOT/page-kv/experiments/logs
mkdir -p "$OUT" "$LOG"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate page-repro

BUDGETS_FULL=0.0625,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0
TASKS=niah_multikey_3,vt,fwe,qa_1

# gated_eviction.py takes --gpu, but pinning via CUDA_VISIBLE_DEVICES keeps
# each shard on its own card regardless of what the script does internally.
cell () {  # cell <gpu> <tag> <model> <config> <extra args...>
  local gpu=$1 tag=$2 model=$3 cfg=$4; shift 4
  echo ">> [gpu$gpu] $tag"
  CUDA_VISIBLE_DEVICES=$gpu python "$SRC/scripts/gated_eviction.py" \
    --model "$model" --config "$cfg" --tasks "$TASKS" \
    --tau 0.07 --score_policy snapkv --gpu 0 \
    --out "$OUT/$tag.jsonl" "$@" > "$LOG/$tag.log" 2>&1 &
}

case "${1:-all}" in

gate)
  echo "=== reproduction gate: Qwen2.5-1.5B 4K, b in {1.0,0.25}, N=100 ==="
  CUDA_VISIBLE_DEVICES=0 python "$SRC/scripts/gated_eviction.py" \
    --model Qwen/Qwen2.5-1.5B-Instruct --config 4096 --tasks "$TASKS" \
    --max_examples 100 --budgets 1.0,0.25 --tau 0.07 --score_policy snapkv \
    --gpu 0 --out "$OUT/repro_qwen15b_4k.jsonl" 2>&1 | tail -5
  python "$ROOT/page-kv/experiments/gpu/repro_diff.py" \
    "$OUT/repro_qwen15b_4k.jsonl" "$SRC/results/gated_4k_qwen15b.jsonl"
  ;;

cheap)
  # Prefill-only work: no decoding, so these are minutes not hours.
  echo "=== P3-2 held-out gate-signal ablation (3 cells, parallel) ==="
  CUDA_VISIBLE_DEVICES=0 python "$SRC/scripts/gate_signal_ablation.py" \
    --model Qwen/Qwen2.5-14B-Instruct --config 4096 \
    --out "$OUT/signal_ablation_qwen14b_4k.jsonl" \
    > "$LOG/p32_qwen14b.log" 2>&1 &
  CUDA_VISIBLE_DEVICES=1 python "$SRC/scripts/gate_signal_ablation.py" \
    --model meta-llama/Llama-3.1-8B-Instruct --config 4096 \
    --out "$OUT/signal_ablation_llama31_4k.jsonl" \
    > "$LOG/p32_llama31.log" 2>&1 &
  CUDA_VISIBLE_DEVICES=2 python "$SRC/scripts/gate_signal_ablation.py" \
    --model mistralai/Mistral-7B-Instruct-v0.3 --config 16384 \
    --out "$OUT/signal_ablation_mistral7b_16k.jsonl" \
    > "$LOG/p32_mistral16k.log" 2>&1 &
  wait
  echo "=== P3-2 done; see $LOG/p32_*.log ==="
  ;;

seeds)
  echo "=== P3-1 seed replicates (seeds 2,3 x 4 cells) ==="
  # Scoping follows the per-task decomposition: Delta is nonzero almost only
  # on NIAH-MK3, so that is where seed variance can actually move a headline
  # number. Full 4-task suite is run so the no-MK3 column gets seeds too.
  for seed in 2 3; do
    i=0
    for spec in "qwen15b:Qwen/Qwen2.5-1.5B-Instruct" \
                "qwen3b:Qwen/Qwen2.5-3B-Instruct" \
                "qwen14b:Qwen/Qwen2.5-14B-Instruct" \
                "mistral7b:mistralai/Mistral-7B-Instruct-v0.3"; do
      tag=${spec%%:*}; model=${spec#*:}
      echo ">> [gpu$i] seed$seed $tag"
      SHUFFLE_SEED=$seed CUDA_VISIBLE_DEVICES=$i \
      PAGE_SRC="$SRC/scripts" python "$ROOT/page-kv/experiments/scripts/gated_eviction_seed.py" \
        --model "$model" --config 4096 --tasks "$TASKS" --max_examples 100 \
        --budgets "$BUDGETS_FULL" --tau 0.07 --score_policy snapkv --gpu 0 \
        --out "$OUT/gated_4k_${tag}_seed${seed}.jsonl" \
        > "$LOG/seed${seed}_${tag}.log" 2>&1 &
      i=$((i+1))
    done
    wait
    echo "=== seed $seed complete ==="
  done
  ;;

all)
  "$0" gate && "$0" cheap && "$0" seeds
  ;;

*) echo "usage: $0 {gate|cheap|seeds|all}"; exit 2 ;;
esac
echo "RUN-P3-DONE:${1:-all}"