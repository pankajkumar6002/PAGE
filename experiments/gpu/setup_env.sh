#!/usr/bin/env bash
# Build a conda env matching the paper's reproducibility statement:
# PyTorch 2.11, Transformers 5.9, Python 3.13.
set -euo pipefail
ENV=page-repro
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | grep -q "^${ENV} " || conda create -y -n "$ENV" python=3.13
conda activate "$ENV"
pip install --quiet "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu130
pip install --quiet "transformers==5.9.0" "datasets" "accelerate" "huggingface_hub"
python - <<'PY'
import torch, transformers, sys
print("python      ", sys.version.split()[0])
print("torch       ", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count(), "gpus")
print("transformers", transformers.__version__)
from transformers.cache_utils import DynamicCache
print("DynamicCache import OK")
PY
