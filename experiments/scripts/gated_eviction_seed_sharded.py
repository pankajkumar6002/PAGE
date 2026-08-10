"""Seeded gated-eviction run with the model sharded across all visible GPUs.

Mistral-7B at 4K with full eager attention peaks above 48 GiB, so it does not
fit one Ada card. Sharding the layers with device_map="auto" pools the visible
cards; the per-layer computation is unchanged, only where each layer's weights
live differs.

Two things this does NOT fix, both worth stating before the numbers are used:

  1. Sharding is expected to be numerically faithful, since layers are computed
     identically and only moved between devices, but "expected" is not
     "measured". Validate against a single-card run of a model that fits before
     treating sharded numbers as equivalent.
  2. Sharding does not remove the host difference. Seeds measured here are on
     RTX 6000 Ada; seeds 1 and 2 of this cell were measured on A100. The
     reproduction gate put that offset at ~8% of generations and 0.0036 on
     mean D, so a seed measured here is a cross-host datapoint, not a
     replicate of those. To compare seeds, all seeds must share a host.
"""
import os
import sys

import torch

_src = os.environ.get("PAGE_SRC")
if _src:
    sys.path.insert(0, _src)
else:
    _here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.abspath(os.path.join(_here, "..", "..", "..", "page-kv",
                                        "experiments", "scripts"))
    if os.path.exists(os.path.join(cand, "gated_eviction.py")):
        sys.path.insert(0, cand)
    else:
        raise SystemExit("set PAGE_SRC to the dir holding gated_eviction.py")

import gated_eviction as ge  # noqa: E402
from datasets import load_dataset as _real_load  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

SEED = int(os.environ["SHUFFLE_SEED"])
_real_fp = AutoModelForCausalLM.from_pretrained


def patched_load_dataset(path, config=None, split=None, **kw):
    return _real_load(path, config, split=split, **kw).shuffle(seed=SEED)


class _Sharded:
    @staticmethod
    def from_pretrained(*a, **kw):
        kw.pop("device_map", None)
        n = torch.cuda.device_count()
        model = _real_fp(*a, device_map=("auto" if n > 1 else {"": 0}), **kw)
        orig_to = model.to

        def _to(*args, **kwargs):
            # the caller chains .to("cuda:N"); honouring it would undo the
            # sharding and reintroduce the OOM
            if args and isinstance(args[0], (str, torch.device)):
                return model
            return orig_to(*args, **kwargs)

        model.to = _to
        return model


if __name__ == "__main__":
    ge.load_dataset = patched_load_dataset
    ge.AutoModelForCausalLM = _Sharded
    print(f"[sharded seed] SHUFFLE_SEED={SEED} "
          f"gpus={torch.cuda.device_count()}", flush=True)
    ge.main()
