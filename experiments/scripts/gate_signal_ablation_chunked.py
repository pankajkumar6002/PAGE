"""v3: bound the attention transient by chunking queries, without touching the math.

Two earlier attempts and what they taught:

  v1 (gate_signal_ablation_retain.py) calls the stock eager attention and
     slices the returned tensor. Proven faithful: 0.0412 vs 0.0412 against the
     stock script on the same host. Bounds retention (48 layers of [H,T,T])
     but not the per-layer transient, so it handles Qwen-14B at 4K and fails
     on Mistral at 16K, where one transient is ~60 GiB.

  v2 (gate_signal_ablation_lowmem.py) recomputed attention with SDPA for the
     output and a separate observation-window matmul for the statistic. It is
     NOT faithful: D drops 0.0448 -> 0.0300, a 33% shift, reproducibly on both
     hosts. Fixing the mask handling did not change it, so the cause is the
     SDPA output path perturbing hidden states, which compounds across layers.
     Do not use it.

v3 keeps the original function as the only thing that computes attention, and
merely feeds it queries in chunks. Attention rows are independent given all
keys, so concatenating per-chunk outputs is exact; the peak transient becomes
[H, CHUNK, Tk] instead of [H, Tq, Tk]. At CHUNK=1024 and T=20k that is ~2.6 GiB
rather than ~60 GiB.

Faithfulness is asserted, not assumed: validate against the stock script on a
cell that fits before trusting it on one that does not.
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
CHUNK = int(os.environ.get("Q_CHUNK", "1024"))
_real_fp = AutoModelForCausalLM.from_pretrained


def _patch():
    import transformers.models.qwen2.modeling_qwen2 as qwen2
    mods = [qwen2]
    for name in ("llama", "mistral"):
        try:
            mods.append(__import__(f"transformers.models.{name}.modeling_{name}",
                                   fromlist=["x"]))
        except Exception:
            pass

    for mod in mods:
        if not hasattr(mod, "eager_attention_forward"):
            continue
        orig = mod.eager_attention_forward

        def patched(module, query, key, value, attention_mask, scaling,
                    dropout=0.0, _orig=orig, **kw):
            Tq = query.shape[2]
            if Tq <= CHUNK:
                out, attn = _orig(module, query, key, value, attention_mask,
                                  scaling, dropout=dropout, **kw)
                if attn is not None and attn.dim() == 4 and attn.shape[2] > OBS:
                    attn = attn[:, :, -OBS:, :].contiguous()
                return out, attn

            outs, tail = [], None
            for i in range(0, Tq, CHUNK):
                j = min(i + CHUNK, Tq)
                q = query[:, :, i:j, :]
                # slice the mask only when it carries a query axis
                m = attention_mask
                if m is not None and m.dim() == 4 and m.shape[-2] == Tq:
                    m = m[..., i:j, :]
                o, a = _orig(module, q, key, value, m, scaling,
                             dropout=dropout, **kw)
                outs.append(o)
                if a is not None and j > Tq - OBS:      # chunk holds obs rows
                    take = a[:, :, -(j - max(i, Tq - OBS)):, :]
                    tail = take if tail is None else torch.cat([tail, take], 2)
                del a
            # eager_attention_forward returns [B, Tq, H, d]
            out = torch.cat(outs, dim=1)
            return out, tail

        mod.eager_attention_forward = patched
        if hasattr(mod, "ALL_ATTENTION_FUNCTIONS"):
            try:
                mod.ALL_ATTENTION_FUNCTIONS.register("eager", patched)
            except Exception:
                pass


class _Loader:
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
    _patch()
    gsa.AutoModelForCausalLM = _Loader
    print(f"[chunked] obs={OBS} q_chunk={CHUNK} gpus={torch.cuda.device_count()}",
          flush=True)
    gsa.main()
