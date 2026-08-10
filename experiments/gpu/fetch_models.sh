#!/usr/bin/env bash
# Pull every checkpoint the P3 runs need. Idempotent; skips what is cached.
set -uo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate page-repro
python - <<'PY'
from huggingface_hub import snapshot_download
import time
MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",   # P3-1 seeds, P3-2/3 probes
    "Qwen/Qwen2.5-3B-Instruct",     # P3-1 seeds
    "mistralai/Mistral-7B-Instruct-v0.3",  # P3-1 seeds, P3-2 held-out 16K
    "Qwen/Qwen2.5-14B-Instruct",    # P3-1 seeds, P3-2 held-out 4K
    "Qwen/Qwen3-4B-Instruct-2507",  # P3-4 DynamicKV head-to-head
]
for m in MODELS:
    t = time.time()
    try:
        snapshot_download(m); print(f"OK   {m}  ({time.time()-t:.0f}s)", flush=True)
    except Exception as e:
        print(f"FAIL {m}: {type(e).__name__}: {e}", flush=True)
PY
echo "FETCH-DONE"
