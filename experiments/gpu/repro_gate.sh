#!/usr/bin/env bash
# Reproduction gate. Everything downstream depends on this.
#
# Re-runs a slice of a cell whose results are already released, under the
# environment that matches the paper's reproducibility statement, and diffs
# row for row against the logged jsonl.
#
# Why this must pass before any seed run: P3-1 measures how much Delta moves
# when the input draw changes. If the environment itself moves Delta, the two
# effects are confounded and the seed numbers mean nothing. A clean diff here
# licenses attributing later differences to the seed.
set -uo pipefail

ROOT=/home/pankaj/Work/PAGE
SRC=$ROOT/page-kv/experiments
OUT=$ROOT/page-kv/experiments/results
mkdir -p "$OUT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate page-repro

export CUDA_VISIBLE_DEVICES=0   # GPUs 1-3 belong to another user

echo "== environment =="
python - <<'PY'
import torch, transformers, sys
print("python      ", sys.version.split()[0])
print("torch       ", torch.__version__)
print("transformers", transformers.__version__)
print("visible gpus", torch.cuda.device_count())
PY

# Two budgets is enough to exercise both arms: b=1.0 is the full-cache
# reference the gate falls back to, b=0.25 is a real eviction point.
echo
echo "== rerunning Qwen2.5-1.5B 4K, 4 tasks, b in {1.0, 0.25}, N=100 =="
time python "$SRC/scripts/gated_eviction.py" \
  --model Qwen/Qwen2.5-1.5B-Instruct --config 4096 \
  --tasks niah_multikey_3,vt,fwe,qa_1 --max_examples 100 \
  --budgets 1.0,0.25 --tau 0.07 --score_policy snapkv \
  --gpu 0 --out "$OUT/repro_qwen15b_4k.jsonl"

echo
echo "== diff against the released log =="
python "$ROOT/page-kv/experiments/gpu/repro_diff.py" \
  "$OUT/repro_qwen15b_4k.jsonl" \
  "$SRC/results/gated_4k_qwen15b.jsonl"
echo "REPRO-GATE-EXIT=$?"