#!/usr/bin/env bash
# Llama architecture bias control + Mistral re-run after the double-BOS fix.
#
# Llama-3.1-8B has (L=32, Q=32, KV=8), identical in shape to Mistral-7B but a
# different family, so it separates "8 KV-heads" from "Mistral-specific" as the
# cause of the per-head Ada-KV P1 violation. Prereg: llama_architecture_bias_prereg.md,
# sha256 1dd0094de0b1bc0b...
#
# Mistral is re-run because the double-BOS bug flipped 18/400 of its gate calls,
# and that is the cell where P1 failed.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R2="$(cd "$HERE/../.." && pwd)"
S="$R2/experiments/scripts"; OUT="$R2/experiments/results"; L="$R2/experiments/logs"
export PAGE_SRC="$R2/experiments/scripts"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate page-repro

python - <<PY || exit 1
import hashlib, sys
p="$R2/preregistration/llama_architecture_bias_prereg.md"
want="1dd0094de0b1bc0bfe2dfd488032a732d065084628b366ba2a14a415350245b0"
got=hashlib.sha256(open(p,"rb").read()).hexdigest()
if got!=want: sys.exit(f"prereg hash mismatch\n  expected {want}\n  got      {got}")
print("prereg verified:", got[:16], "...")
PY

T=niah_multikey_3,vt,fwe,qa_1
B=0.0625,0.125,0.25,0.375,0.5,0.625,0.75,0.875
GPU=${GPU:-0}
run(){ CUDA_VISIBLE_DEVICES=$GPU python "$S/adakv_matrix.py" --model "$2" --slug "$1" \
  --config 4096 --tasks "$T" --max_examples 100 --budgets "$B" --tau 0.07 \
  --floor_alpha 0.5 --out "$OUT/adakv_4k_$1.jsonl" > "$L/adakv_$1.log" 2>&1
  echo "  $1 exit=$? rows=$(wc -l < "$OUT/adakv_4k_$1.jsonl" 2>/dev/null || echo 0)"; }

date -u +"start: %FT%TZ"
echo "=== [1/2] architecture-bias control: Llama-3.1-8B (8 KV-heads, different family) ==="
run llama31 meta-llama/Llama-3.1-8B-Instruct
echo "=== [2/2] per-head Ada-KV re-run: Mistral-7B (post double-BOS fix) ==="
run mistral7b mistralai/Mistral-7B-Instruct-v0.3
date -u +"end: %FT%TZ"
