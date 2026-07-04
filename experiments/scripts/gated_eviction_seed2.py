"""Second-seed (different input draw) wrapper around gated_eviction.main().

gated_eviction.py selects the first `max_examples` inputs per task
deterministically. This wrapper monkeypatches its load_dataset so the full
RULER split is SHUFFLED with SHUFFLE_SEED (env) first; gated_eviction then
takes the first `max_examples` of the shuffled order per task, i.e. a genuinely
different input draw. All model/eviction/gate logic is unchanged.
"""
import os
import sys

sys.path.insert(0, "/home/smlab/projects/eff-nn/experiments/scripts")
import gated_eviction
from datasets import load_dataset as _real_load_dataset

SEED = int(os.environ["SHUFFLE_SEED"])


def patched_load_dataset(path, config=None, split=None, **kw):
    ds = _real_load_dataset(path, config, split=split, **kw)
    return ds.shuffle(seed=SEED)


gated_eviction.load_dataset = patched_load_dataset

if __name__ == "__main__":
    print(f"[gated seed2 wrapper] SHUFFLE_SEED={SEED}")
    gated_eviction.main()
