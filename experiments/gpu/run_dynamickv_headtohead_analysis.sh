#!/usr/bin/env bash
# DynamicKV adaptive-budget head-to-head, extra models.
#
# The headline already holds on the released Qwen3-4B run
# (dynamickv_qwen3_4k.jsonl): P1 (adaptive alloc does NOT rescue MK3) and P2
# (gate recovers +1.000) both hold; dynamickv_headtohead_analysis.py CHECK: PASS. This
# script broadens the claim to more models via the SAME runner, unchanged.
#
# Runner facts (verified): single card (--gpu, cuda:{gpu}, eager attn, .to(dev)
# -- NO sharding, so the model must fit one card), RULER data only
# (ex["answer"]). CLI: --model --config --tasks --max_examples --budgets --tau
# --gpu --out.
#
# NOT auto-queued. Launch only when the user says a GPU is free.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R2="$(cd "$HERE/../.." && pwd)"
OUT="$R2/experiments/results"
LOGDIR="$R2/experiments/logs"
export PAGE_SRC="$R2/experiments/scripts"
GPU="${1:-0}"

mkdir -p "$OUT" "$LOGDIR"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate page-repro

# Gate on the pre-registration hash.
python - <<PY || exit 1
import hashlib, sys
p = "$R2/preregistration/dynamickv_headtohead_prereg.md"
raw = open(p, "rb").read()
i = raw.find(b"## Addendum")
frozen = raw[:i-6] if i != -1 else raw
got = hashlib.sha256(frozen).hexdigest()
want = open(p.replace(".md", ".sha256")).read().split()[0]
if got != want:
    sys.exit(f"prereg hash mismatch\n  want {want}\n  got  {got}")
print("[gate] prereg intact:", got[:16])
PY

# Models that fit one 80GB card in eager one-pass at 4K. Qwen2.5-14B does NOT
# (it needs sharding, which this runner does not do), so it is excluded here;
# use Qwen2.5-3B and Mistral-7B to broaden beyond the released Qwen3-4B cell.
for SPEC in "Qwen/Qwen2.5-3B-Instruct:qwen3b" "mistralai/Mistral-7B-Instruct-v0.3:mistral7b"; do
  MODEL="${SPEC%%:*}"; SLUG="${SPEC##*:}"
  echo "[dynamickv] launching DynamicKV head-to-head for $SLUG on GPU $GPU"
  python "$PAGE_SRC/dynamickv_headtohead.py" \
    --model "$MODEL" --config 4096 \
    --tasks "niah_multikey_3,vt,fwe,qa_1" \
    --max_examples 100 --budgets "1.0,0.5,0.25,0.125,0.0625" \
    --tau 0.07 --gpu "$GPU" \
    --out "$OUT/dynamickv_${SLUG}_4k.jsonl" \
    > "$LOGDIR/dynamickv_headtohead_${SLUG}.log" 2>&1

  echo "[dynamickv] analyzing $SLUG"
  python "$PAGE_SRC/dynamickv_headtohead_analysis.py" \
    "$OUT/dynamickv_${SLUG}_4k.jsonl" \
    >> "$LOGDIR/dynamickv_headtohead_${SLUG}.log" 2>&1
done

echo "[dynamickv] extra-model runs done. dynamickv_headtohead_analysis.py can take all logs"
echo "            at once (multiple paths) to produce a combined per-model table."
