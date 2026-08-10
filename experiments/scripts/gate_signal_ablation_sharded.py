"""Sharded launcher for gate_signal_ablation.py.

page-kv's gate_signal_ablation.py loads with `.to(f"cuda:{gpu}")`, pinning the
whole model to one card. Qwen2.5-14B with eager attention needs more than a
48 GiB card holds (observed: 45.5 GiB resident, then OOM allocating a further
2.44 GiB), so the held-out cell cannot run single-GPU on the Ada host.

This wrapper patches only the model construction to use accelerate's
device_map, spreading the weights over the visible GPUs. Nothing about the
signal computation changes: the ablation reads per-head prefill attention and
computes the same statistics, so results are comparable to the single-GPU
cells. Set PAGE_SRC to the directory holding gate_signal_ablation.py.

  CUDA_VISIBLE_DEVICES=0,1 python gate_signal_ablation_sharded.py \
      --model Qwen/Qwen2.5-14B-Instruct --config 4096 --out ...
"""
import os
import sys

import torch

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
        if os.path.exists(os.path.join(cand, "gate_signal_ablation.py")):
            sys.path.insert(0, cand)
            break
    else:
        raise SystemExit("set PAGE_SRC to the dir containing "
                         "gate_signal_ablation.py")

import gate_signal_ablation as gsa  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

_real_from_pretrained = AutoModelForCausalLM.from_pretrained


class _Sharded:
    """from_pretrained that shards, and whose .to(device) is a no-op.

    The original script chains `.to(device)` onto the load. Moving an
    already-dispatched sharded model would undo the sharding and OOM, so the
    returned object absorbs that call.
    """

    @staticmethod
    def from_pretrained(*a, **kw):
        kw.pop("device_map", None)
        model = _real_from_pretrained(*a, device_map="auto", **kw)
        orig_to = model.to

        def _to(*args, **kwargs):
            if args and isinstance(args[0], (str, torch.device)):
                return model            # ignore the pin-to-one-card request
            return orig_to(*args, **kwargs)

        model.to = _to
        return model


if __name__ == "__main__":
    gsa.AutoModelForCausalLM = _Sharded
    # the script calls torch.cuda.set_device(args.gpu); harmless under
    # device_map, but keep it pointing at a card that exists
    n = torch.cuda.device_count()
    print(f"[sharded] visible GPUs: {n}", flush=True)
    gsa.main()
