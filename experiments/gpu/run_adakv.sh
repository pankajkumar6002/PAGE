#!/usr/bin/env bash
# Per-head Ada-KV plain arm for the headline matrix.
#
# One model per card, four cards, all four cells in parallel. Follows the prior
# round's run_stage.sh pattern: conda page-repro, PAGE_SRC exported, one log per
# cell under ../../logs/.
#
# The pre-registration must be intact before anything launches.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R2="$(cd "$HERE/../.." && pwd)"
SCRIPTS="$R2/experiments/scripts"
OUT="$R2/experiments/results"
LOGDIR="$R2/experiments/logs"
export PAGE_SRC="$R2/experiments/scripts"

mkdir -p "$OUT" "$LOGDIR"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate page-repro

# Gate: refuse to run if the predictions were edited after hashing.
python - <<PY || exit 1
import sys
sys.path.insert(0, "$SCRIPTS")
from paths import verify_prereg
print("prereg sha256 verified:", verify_prereg()[:16], "...")
PY

TASKS=niah_multikey_3,vt,fwe,qa_1
BUDGETS=0.0625,0.125,0.25,0.375,0.5,0.625,0.75,0.875
N=${N:-100}

# slug:hf_id:cuda_devices:extra_args
# Qwen2.5-14B needs two cards: at 4K it materialises L*H*T*T*2 = 64.4 GB of
# attention tensors plus ~28 GB of weights, which does not fit one 80 GB card
# in one-pass mode. Sharding keeps it on the SAME one-pass path as the other
# three cells; --two_pass would change what the scorer sees, which is exactly
# the H2O/SnapKV degeneracy documented in h2o_degeneracy_audit.md.
CELLS=(
  "qwen15b:Qwen/Qwen2.5-1.5B-Instruct:0:"
  "qwen3b:Qwen/Qwen2.5-3B-Instruct:1:"
  "mistral7b:mistralai/Mistral-7B-Instruct-v0.3:2:"
  "qwen14b:Qwen/Qwen2.5-14B-Instruct:3,0:--device_map auto"
)

echo "=== Ada-KV matrix: launching ${#CELLS[@]} cells, N=$N ==="
date -u +"start: %Y-%m-%dT%H:%M:%SZ"

pids=()
for i in "${!CELLS[@]}"; do
  IFS=':' read -r slug model devs extra <<< "${CELLS[$i]}"
  log="$LOGDIR/adakv_${slug}.log"
  echo "  GPU $devs -> $slug ($model) $extra  log=$log"
  CUDA_VISIBLE_DEVICES="$devs" python "$SCRIPTS/adakv_matrix.py" \
    --model "$model" --slug "$slug" --config 4096 \
    --tasks "$TASKS" --max_examples "$N" --budgets "$BUDGETS" \
    --tau 0.07 --floor_alpha 0.5 $extra \
    --out "$OUT/adakv_4k_${slug}.jsonl" > "$log" 2>&1 &
  pids+=($!)
done

echo "waiting on ${pids[*]}"
fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    slug="${CELLS[$i]%%:*}"
    echo "FAILED: $slug (see $LOGDIR/adakv_${slug}.log)"
    fail=1
  fi
done

date -u +"end: %Y-%m-%dT%H:%M:%SZ"
for i in "${!CELLS[@]}"; do
  slug="${CELLS[$i]%%:*}"
  f="$OUT/adakv_4k_${slug}.jsonl"
  printf "  %-12s %s rows\n" "$slug" "$([ -f "$f" ] && wc -l < "$f" || echo MISSING)"
done

if [ "$fail" -eq 0 ]; then
  echo "ALL CELLS COMPLETED"
else
  echo "SOME CELLS FAILED - do not promote numbers into the paper"
fi
exit $fail