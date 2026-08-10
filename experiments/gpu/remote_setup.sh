#!/usr/bin/env bash
# One-shot bring-up of the PAGE run environment on a bare host.
#
# Installs miniconda if absent (the host has only system python 3.12; the
# paper's reproducibility statement pins 3.13), builds the pinned env, and
# pulls every checkpoint the P3 runs need. Idempotent: safe to re-run.
set -uo pipefail

ROOT=$HOME/Work/PAGE
ENV=page-repro
MC=$HOME/miniconda3

echo "===== 1. conda ====="
if [ ! -x "$MC/bin/conda" ]; then
  echo ">> installing miniconda to $MC"
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh
  bash /tmp/mc.sh -b -p "$MC" >/dev/null
  rm -f /tmp/mc.sh
fi
source "$MC/etc/profile.d/conda.sh"

echo "===== 2. env $ENV (python 3.13) ====="
conda env list | grep -q "^${ENV} " || conda create -y -n "$ENV" python=3.13 >/dev/null
conda activate "$ENV"

python -c "import torch" 2>/dev/null || {
  echo ">> installing torch 2.11 (cu130)"
  pip install --quiet "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu130
}
pip install --quiet "transformers==5.9.0" datasets accelerate huggingface_hub

python - <<'PY'
import sys, torch, transformers
from transformers.cache_utils import DynamicCache
print("python      ", sys.version.split()[0])
print("torch       ", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.device_count(), "gpus")
print("transformers", transformers.__version__)
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  gpu{i}: {p.name}  {p.total_memory/2**30:.0f} GiB")
PY

echo "===== 3. models ====="
python - <<'PY'
from huggingface_hub import snapshot_download
import time
for m in ["Qwen/Qwen2.5-1.5B-Instruct",
          "Qwen/Qwen2.5-3B-Instruct",
          "mistralai/Mistral-7B-Instruct-v0.3",
          "Qwen/Qwen2.5-14B-Instruct",
          "meta-llama/Llama-3.1-8B-Instruct",
          "Qwen/Qwen3-4B-Instruct-2507"]:
    t = time.time()
    try:
        snapshot_download(m)
        print(f"OK   {m}  ({time.time()-t:.0f}s)", flush=True)
    except Exception as e:
        print(f"FAIL {m}: {type(e).__name__}: {e}", flush=True)
PY

echo "===== 4. RULER dataset ====="
python - <<'PY'
from datasets import load_dataset
for cfg in ("4096", "16384"):
    try:
        d = load_dataset("simonjegou/ruler", cfg, split="test")
        print(f"OK   ruler/{cfg}: {len(d)} rows", flush=True)
    except Exception as e:
        print(f"FAIL ruler/{cfg}: {type(e).__name__}: {e}", flush=True)
PY

echo "SETUP-DONE"