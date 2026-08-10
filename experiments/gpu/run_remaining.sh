#!/usr/bin/env bash
# E8 remaining cells, sequential. Only GPUs 0 and 2 are free (1 and 3 belong to
# other users), so qwen14b takes both cards to itself first -- it OOMed once
# when it had to share GPU 0 -- and qwen3b follows on a single card.
set -uo pipefail
R2=/home/pankaj/Work/PAGE/page-kv
SCRIPTS="$R2/experiments/scripts"; OUT="$R2/experiments/results"; LOGDIR="$R2/experiments/logs"
export PAGE_SRC=/home/pankaj/Work/PAGE/page-kv/experiments/scripts
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate page-repro
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python - <<'PY' || exit 1
import sys; sys.path.insert(0,"/home/pankaj/Work/PAGE/page-kv/experiments/scripts")
from paths import verify_prereg; print("prereg verified:", verify_prereg()[:16], "...")
PY

TASKS=niah_multikey_3,vt,fwe,qa_1
BUDGETS=0.0625,0.125,0.25,0.375,0.5,0.625,0.75,0.875

echo "=== [1/2] qwen14b sharded on GPUs 0,2 ==="; date -u +"start: %FT%TZ"
CUDA_VISIBLE_DEVICES=0,2 python "$SCRIPTS/adakv_matrix.py" \
  --model Qwen/Qwen2.5-14B-Instruct --slug qwen14b --config 4096 \
  --tasks "$TASKS" --max_examples 100 --budgets "$BUDGETS" \
  --tau 0.07 --floor_alpha 0.5 --device_map auto \
  --out "$OUT/adakv_4k_qwen14b.jsonl" > "$LOGDIR/adakv_qwen14b.log" 2>&1
echo "qwen14b exit=$?"

# [2/2] qwen3b was started separately on GPU 2 while 14B was still running,
# since GPU 2 had ~63 GB free. Running it here too would overwrite that run
# mid-flight, so this stage is disabled. Re-enable only if the GPU-2 run died.
echo "=== [2/2] qwen3b SKIPPED - already running on GPU 2 ==="
if [ ! -s "$OUT/adakv_4k_qwen3b.jsonl" ]; then
  echo "WARNING: no qwen3b output found; the GPU-2 run may have failed."
fi
date -u +"end: %FT%TZ"
for s in qwen15b qwen3b qwen14b mistral7b; do
  f="$OUT/adakv_4k_$s.jsonl"
  printf "  %-12s %s rows\n" "$s" "$([ -f "$f" ] && wc -l < "$f" || echo MISSING)"
done
