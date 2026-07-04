"""Second-seed (different input draw) wrapper around longbench_capkv.main().

longbench_capkv.py takes the first `max_examples` LongBench examples. This
wrapper shuffles the split with SHUFFLE_SEED (env) first, so a different draw
of examples is evaluated. All CapKV-proxy / gating logic is unchanged.
"""
import os
import sys

sys.path.insert(0, "/home/smlab/projects/eff-nn/experiments/scripts")
import longbench_capkv
from datasets import load_dataset as _real_load_dataset

SEED = int(os.environ["SHUFFLE_SEED"])


def patched_load_dataset(path, config=None, split=None, **kw):
    ds = _real_load_dataset(path, config, split=split, **kw)
    return ds.shuffle(seed=SEED)


longbench_capkv.load_dataset = patched_load_dataset

if __name__ == "__main__":
    print(f"[capkv seed2 wrapper] SHUFFLE_SEED={SEED}")
    longbench_capkv.main()
