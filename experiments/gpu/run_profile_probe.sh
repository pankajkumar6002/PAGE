#!/usr/bin/env bash
# E12 (Q2 / W4): OPTIONAL id-tagged profile re-dump to harden the probe.
#
# The E12 headline runs OFFLINE on released drops_*.jsonl (profile_probe_ablation.py),
# and its result is already computed (NO IMPROVEMENT: probe 0.571 < raw-D 0.741
# on Llama). This script only produces a hardened, id-tagged profile dump so the
# probe can be re-fit with more inputs and an explicit (task,id) label, and to
# add the in-architecture variant flagged in the prereg addendum (a probe fit on
# Llama's OWN pilot rather than transferred from Qwen).
#
# NOT auto-queued. Launch only when the user says a GPU is free. Prefill-only,
# minutes per 100 inputs.
set -uo pipefail

R2=/home/pankaj/Work/PAGE/new-exp-page-kv-r2
OUT="$R2/experiments/results"
LOGDIR="$R2/logs"
export PAGE_SRC=/home/pankaj/Work/PAGE/page-kv/experiments/scripts
GPU="${1:-0}"

mkdir -p "$OUT" "$LOGDIR"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate page-repro

# Gate on the E12 pre-registration hash.
python - <<'PY' || exit 1
import hashlib, sys
p = "/home/pankaj/Work/PAGE/new-exp-page-kv-r2/preregistration/profile_probe_ablation_prereg.md"
raw = open(p, "rb").read()
i = raw.find(b"## Addendum")
frozen = raw[:i-6] if i != -1 else raw
got = hashlib.sha256(frozen).hexdigest()
want = open(p.replace(".md", ".sha256")).read().split()[0]
if got != want:
    sys.exit(f"E12 prereg hash mismatch\n  want {want}\n  got  {got}")
print("[gate] E12 prereg intact:", got[:16])
PY

# per_layer_agreement_probe.py already dumps head_agreement_per_layer but WITHOUT
# an id field. We add id by post-processing: the probe emits rows in
# task-then-selection order, so an enumerate-per-task id reproduces the same
# positional index the eviction logs use (verified: profile-derived D == logged
# D exactly, per profile_probe_ablation.py's guard). This wrapper runs the dump then
# stamps ids.
for SPEC in "Qwen/Qwen2.5-1.5B-Instruct:qwen15b" "meta-llama/Llama-3.1-8B-Instruct:llama31"; do
  MODEL="${SPEC%%:*}"; SLUG="${SPEC##*:}"
  RAW="$OUT/profiles_${SLUG}_4k.jsonl"
  echo "[e12] dumping per-layer profiles for $SLUG on GPU $GPU"
  python "$PAGE_SRC/per_layer_agreement_probe.py" \
    --model "$MODEL" --config 4096 \
    --tasks "niah_multikey_3,vt,fwe,qa_1,niah_multivalue" \
    --n_per_task 100 --gpu "$GPU" \
    --out "$RAW" \
    > "$LOGDIR/e12_profiles_${SLUG}.log" 2>&1

  # stamp per-(task,selection-order) id so downstream joins are explicit.
  python - "$RAW" <<'PY'
import json, sys
from collections import defaultdict
path = sys.argv[1]
try:
    rows = [json.loads(l) for l in open(path) if l.strip()]
except OSError as e:
    sys.exit(f"cannot read {path}: {e}")
seen = defaultdict(int)
for r in rows:
    t = r["task"]
    r["id"] = seen[t]
    seen[t] += 1
with open(path, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"[e12] stamped ids on {len(rows)} rows of {path}")
PY
done

echo "[e12] id-tagged profile dumps written. Re-run profile_probe_ablation.py pointed at"
echo "      profiles_*_4k.jsonl (via PAGE_RESULTS override) to re-fit with"
echo "      explicit ids, and add the Llama-own-pilot in-architecture variant."