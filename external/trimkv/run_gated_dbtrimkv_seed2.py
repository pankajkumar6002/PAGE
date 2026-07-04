"""Second-seed (different input draw) wrapper around run_gated_dbtrimkv.main().

run_gated_dbtrimkv.py takes the first `per_task` RULER examples per task
deterministically. This wrapper shuffles the RULER split with SHUFFLE_SEED
(env) before that slice, so a genuinely different input draw is evaluated.
All DBTrimKV / gate logic is unchanged.
"""
import os
import sys

# Must be set before run_gated_dbtrimkv imports torch / sets its own default.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("VISIBLE_GPU", "1"))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_gated_dbtrimkv
from datasets import load_dataset as _real_load_dataset

SEED = int(os.environ["SHUFFLE_SEED"])


def patched_load_dataset(path, config=None, split=None, **kw):
    ds = _real_load_dataset(path, config, split=split, **kw)
    return ds.shuffle(seed=SEED)


run_gated_dbtrimkv.load_dataset = patched_load_dataset

if __name__ == "__main__":
    print(f"[dbtrimkv seed2 wrapper] SHUFFLE_SEED={SEED} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    run_gated_dbtrimkv.main()
