"""v1 memory-bounded launcher: retention-only, computation untouched.

Calls the stock eager attention and slices the returned tensor, so the
statistic is whatever the original would have produced. Bounds retention
(48 layers of [H,T,T]) but not the per-layer transient, so it handles
Qwen-14B at 4K but not Mistral at 16K.

Original header follows.


page-kv's gate_signal_ablation.py calls the model with output_attentions=True,
which retains every layer's full [H, T, T] attention matrix. That is fine on
Qwen2.5-1.5B (12 heads, 28 layers) but not on Qwen2.5-14B: at T=4096 one layer
is 2.5 GiB in fp32 and 48 layers is ~120 GiB, so the cell OOMs on an 80 GiB
A100 and on a 48 GiB Ada card alike. Sharding does not help, because the
problem is retention, not weights.

The paper's own gate does not pay this. Section limitations describes a
two-pass mode that scores only the last w observation queries against the
cache, i.e. O(w*T) rather than O(T^2); at w=32 that is 20 MiB per layer
instead of 2.5 GiB, a 128x reduction.

This wrapper applies that shape to the ablation. It replaces the captured
attention with the observation-window slice, computed per layer through a
hook and released immediately, so peak memory is one layer's slice rather than
all layers' full matrices. The statistics the ablation computes read only the
last w query rows anyway, so the numbers are unchanged; what changes is that
the other T-w rows are never materialised.

  CUDA_VISIBLE_DEVICES=0,1 PAGE_SRC=... python gate_signal_ablation_lowmem.py \
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

OBS = int(os.environ.get("OBS_WINDOW", "32"))
_real_fp = AutoModelForCausalLM.from_pretrained


def _patch_attention_modules():
    """Make eager attention emit only the last OBS query rows."""
    import transformers.models.qwen2.modeling_qwen2 as qwen2
    mods = [qwen2]
    for name in ("llama", "mistral"):
        try:
            mods.append(__import__(
                f"transformers.models.{name}.modeling_{name}",
                fromlist=["x"]))
        except Exception:
            pass

    for mod in mods:
        if not hasattr(mod, "eager_attention_forward"):
            continue
        orig = mod.eager_attention_forward

        def patched(module, query, key, value, attention_mask, scaling,
                    dropout=0.0, _orig=orig, **kw):
            # v1: do NOT reimplement attention. Call the original, then keep
            # only the observation rows so the other layers' full matrices are
            # not retained. The computation is byte-for-byte the original's;
            # only retention changes.
            out, attn = _orig(module, query, key, value, attention_mask,
                              scaling, dropout=dropout, **kw)
            if attn is not None and attn.dim() == 4 and attn.shape[2] > OBS:
                attn = attn[:, :, -OBS:, :].contiguous()
            return out, attn

        mod.eager_attention_forward = patched
        if hasattr(mod, "ALL_ATTENTION_FUNCTIONS"):
            try:
                mod.ALL_ATTENTION_FUNCTIONS.register("eager", patched)
            except Exception:
                pass


class _Sharded:
    @staticmethod
    def from_pretrained(*a, **kw):
        kw.pop("device_map", None)
        n = torch.cuda.device_count()
        model = _real_fp(*a, device_map=("auto" if n > 1 else {"": 0}), **kw)
        orig_to = model.to

        def _to(*args, **kwargs):
            if args and isinstance(args[0], (str, torch.device)):
                return model
            return orig_to(*args, **kwargs)

        model.to = _to
        return model


if __name__ == "__main__":
    _patch_attention_modules()
    gsa.AutoModelForCausalLM = _Sharded
    print(f"[lowmem] obs_window={OBS}, visible GPUs={torch.cuda.device_count()}",
          flush=True)
    gsa.main()
