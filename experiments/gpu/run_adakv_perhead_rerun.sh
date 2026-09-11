#!/usr/bin/env bash
# Per-head Ada-KV rerun after the shared-mask fix. Only GPUs 0 and 2 are ours.
# Stage 1: qwen15b (GPU 0) + mistral7b (GPU 2) in parallel.
# Stage 2: qwen3b (GPU 0) + qwen14b sharded (0,2) would contend, so qwen3b
#          runs on GPU 0 and 14B follows alone across both cards.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R2="$(cd "$HERE/../.." && pwd)"
S="$R2/experiments/scripts"; OUT="$R2/experiments/results"; L="$R2/experiments/logs"
export PAGE_SRC="$R2/experiments/scripts"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate page-repro
python -c "import sys;sys.path.insert(0,'$S');from paths import verify_prereg;print('prereg:',verify_prereg()[:16],'...')" || exit 1

T=niah_multikey_3,vt,fwe,qa_1
B=0.0625,0.125,0.25,0.375,0.5,0.625,0.75,0.875
run(){ CUDA_VISIBLE_DEVICES=$2 python "$S/adakv_matrix.py" --model "$3" --slug "$1" \
  --config 4096 --tasks "$T" --max_examples 100 --budgets "$B" --tau 0.07 \
  --floor_alpha 0.5 ${4:-} --out "$OUT/adakv_4k_$1.jsonl" > "$L/adakv_$1.log" 2>&1; echo "  $1 exit=$?"; }

date -u +"start: %FT%TZ"
echo "=== stage 1: qwen15b (GPU0) + mistral7b (GPU2) ==="
run qwen15b 0 Qwen/Qwen2.5-1.5B-Instruct & p1=$!
run mistral7b 2 mistralai/Mistral-7B-Instruct-v0.3 & p2=$!
wait $p1 $p2
echo "=== stage 2: qwen3b (GPU0) ==="
run qwen3b 0 Qwen/Qwen2.5-3B-Instruct
echo "=== stage 3: qwen14b sharded (GPU0,2) ==="
run qwen14b 0,2 Qwen/Qwen2.5-14B-Instruct "--device_map auto"
date -u +"end: %FT%TZ"
for s in qwen15b qwen3b mistral7b qwen14b; do
  printf "  %-11s %s rows\n" "$s" "$([ -f "$OUT/adakv_4k_$s.jsonl" ] && wc -l < "$OUT/adakv_4k_$s.jsonl" || echo MISSING)"; done
