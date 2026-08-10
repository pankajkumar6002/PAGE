"""Seeded input-draw wrapper around gated_eviction.main().

Replaces page-kv/experiments/scripts/gated_eviction_seed2.py, which hardcodes
the original run machine's path (/home/smlab/projects/eff-nn/...) and so does
not import anywhere else.

What the seed varies: gated_eviction.py takes the first `max_examples` inputs
per task deterministically, so shuffling the RULER split before that selection
gives a genuinely different draw. Decoding stays greedy, and the model,
eviction and gate logic are untouched. The seed therefore measures sampling
variance over which inputs are evaluated, which is the quantity a reviewer
asking "is this one lucky draw?" wants, and not decoding randomness.

  SHUFFLE_SEED=2 PAGE_SRC=/path/to/page-kv/experiments/scripts \
      python gated_eviction_seed.py --model ... --out ...
"""
import os
import sys

# Locate the original gated_eviction module without assuming any absolute path.
_src = os.environ.get("PAGE_SRC")
if _src:
    sys.path.insert(0, _src)
else:
    _here = os.path.dirname(os.path.abspath(__file__))
    for cand in (
        os.path.join(_here, "..", "..", "..", "page-kv", "experiments", "scripts"),
        os.path.join(_here, "..", "..", "page-kv", "experiments", "scripts"),
    ):
        cand = os.path.abspath(cand)
        if os.path.exists(os.path.join(cand, "gated_eviction.py")):
            sys.path.insert(0, cand)
            break
    else:
        raise SystemExit(
            "cannot locate gated_eviction.py; set PAGE_SRC to the directory "
            "containing it"
        )

import gated_eviction  # noqa: E402
from datasets import load_dataset as _real_load_dataset  # noqa: E402

SEED = int(os.environ["SHUFFLE_SEED"])


def patched_load_dataset(path, config=None, split=None, **kw):
    return _real_load_dataset(path, config, split=split, **kw).shuffle(seed=SEED)


gated_eviction.load_dataset = patched_load_dataset

if __name__ == "__main__":
    print(f"[seed wrapper] SHUFFLE_SEED={SEED}  gated_eviction from "
          f"{os.path.dirname(gated_eviction.__file__)}", flush=True)
    gated_eviction.main()
