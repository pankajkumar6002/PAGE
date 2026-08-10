"""P3-4: DynamicKV head-to-head against the PAGE gate.

DynamicKV (arXiv 2412.14838) is the closest conceptual neighbour: it observes
that the layer-wise attention pattern differs across task families and turns
that into a *task-aware, per-layer adaptive budget*. PAGE instead turns a
per-input scalar into a *whether-to-evict* decision. The paper argues these
are orthogonal but overlapping; nobody has run them against each other.

IMPLEMENTATION HONESTY. There is no public reference implementation, so this
is a reimplementation from the paper's description and it may differ from the
authors' in details. What is implemented:

  1. per layer, score positions by mean attention from the last obs_window
     queries (the same pooling SnapKV and this codebase already use);
  2. per layer, measure concentration as the share of attention mass held by
     the top-`probe` positions. A layer whose mass is spread out needs more
     budget to retain the same information than one that is already peaked;
  3. distribute the global budget across layers in proportion to that need,
     subject to a floor so no layer is starved, and renormalise so the total
     kept count matches the uniform-budget baseline exactly. The comparison is
     therefore at matched memory, not matched nominal budget;
  4. keep the top-n_l positions per layer, always protecting sinks and recency.

Step 3 is where a reimplementation can most easily diverge from the original,
so the headline comparison is reported as "our DynamicKV-style reimplementation"
rather than as DynamicKV, and the uniform-budget control is reported alongside
so the reader can see what the adaptivity itself buys.

Three arms are run at each budget:
  plain      DynamicKV-style eviction, always on
  gated      PAGE gate wrapped around it (keep full cache when D < tau)
  uniform    same scorer, uniform per-layer budget (isolates the adaptivity)
"""
import argparse
import json
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
from datasets import load_dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402


OBS_SLICE = 32
_STATE = {"active": False, "mask": None}


def _fold_mask(attention_mask, keep, key, query):
    """Fold a [T] keep-set into the additive attention mask."""
    cur = key.shape[-2]
    k = keep
    if cur > k.shape[0]:                       # decoded tokens are always kept
        k = torch.cat([k, torch.ones(cur - k.shape[0], dtype=torch.bool,
                                     device=k.device)])
    elif cur < k.shape[0]:
        k = k[:cur]
    add = torch.zeros(cur, dtype=query.dtype, device=query.device)
    add.masked_fill_(~k, float("-inf"))
    a4 = add[None, None, None, :]
    return a4 if attention_mask is None else attention_mask[..., :cur] + a4


def _patch_retention():
    """Keep only the observation rows of each layer's attention.

    output_attentions=True retains every layer's full [H,T,T]; on Qwen3-4B at
    4K that is ~77 GiB and OOMs any single card. This is the v1 patch already
    validated against the stock scorer (0.0412 vs 0.0412): the original
    attention function still does the computation, we only discard the query
    rows nobody reads. Peak drops to ~0.6 GiB.
    """
    import transformers.models.qwen2.modeling_qwen2 as qwen2
    mods = [qwen2]
    for name in ("qwen3", "llama", "mistral"):
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
            if _STATE["active"]:
                # Adaptive per-layer budgets give each layer a different kept
                # count. A DynamicCache cannot hold ragged layers, so instead
                # of pruning we keep the full cache and mask the evicted keys
                # to -inf. That reproduces exactly the logits a ragged pruned
                # cache would produce, with no padding and no position surgery
                # (the same documented deviation manifoldkv_adakv.py uses).
                keep = _STATE["mask"][module.layer_idx]         # [T] bool
                m = _fold_mask(attention_mask, keep, key, query)
                out, attn = _orig(module, query, key, value, m,
                                  scaling, dropout=dropout, **kw)
            else:
                out, attn = _orig(module, query, key, value, attention_mask,
                                  scaling, dropout=dropout, **kw)
            if attn is not None and attn.dim() == 4 and attn.shape[2] > OBS_SLICE:
                attn = attn[:, :, -OBS_SLICE:, :].contiguous()
            return out, attn

        mod.eager_attention_forward = patched
        if hasattr(mod, "ALL_ATTENTION_FUNCTIONS"):
            try:
                mod.ALL_ATTENTION_FUNCTIONS.register("eager", patched)
            except Exception:
                pass


def layer_scores_and_need(attentions, obs_window, T, device, probe=64):
    """Return per-layer position scores [L,T] and a per-layer need in [0,1]."""
    scores, needs = [], []
    for A in attentions:                       # A: [1,H,q,T]
        a = A[0][:, -obs_window:, :].float().mean(1).mean(0)   # [T]
        scores.append(a)
        k = min(probe, a.numel())
        top = torch.topk(a, k).values.sum()
        # concentration in [0,1]; a diffuse layer has low concentration and,
        # by the argument above, a higher need for budget
        conc = (top / (a.sum() + 1e-9)).clamp(0, 1)
        needs.append(1.0 - conc)
    return torch.stack(scores), torch.stack(needs)


def adaptive_budgets(needs, total_middle, L, floor_frac=0.2):
    """Split `total_middle` kept positions across layers proportional to need."""
    floor = int(floor_frac * total_middle / L)
    free = max(0, total_middle - floor * L)
    w = needs / (needs.sum() + 1e-9)
    alloc = (w * free).round().long() + floor
    # fix rounding so the total matches exactly: matched memory, not matched
    # nominal budget, is what makes the comparison meaningful
    diff = total_middle - int(alloc.sum())
    if diff != 0:
        order = torch.argsort(w, descending=(diff > 0))
        for i in range(abs(diff)):
            alloc[order[i % L]] += 1 if diff > 0 else -1
    return alloc.clamp(min=0)


def masks_from(scores, alloc, T, obs_window, n_sink, device):
    L = scores.shape[0]
    masks = torch.zeros(L, T, dtype=torch.bool, device=device)
    for i in range(L):
        m = masks[i]
        m[:n_sink] = True
        m[T - obs_window:] = True
        lo, hi = n_sink, T - obs_window
        n = int(alloc[i].item())
        if n > 0 and hi > lo:
            cs = scores[i][lo:hi]
            k = min(n, cs.shape[0])
            m[lo + torch.topk(cs, k).indices] = True
    return masks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--config", default="4096")
    p.add_argument("--tasks", default="niah_multikey_3,vt,fwe,qa_1")
    p.add_argument("--max_examples", type=int, default=50)
    p.add_argument("--budgets", default="1.0,0.5,0.25,0.125,0.0625")
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--obs_window", type=int, default=32)
    p.add_argument("--n_sink", type=int, default=4)
    p.add_argument("--top_k", type=int, default=32)
    p.add_argument("--max_new", type=int, default=128)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    global OBS_SLICE
    OBS_SLICE = a.obs_window
    _patch_retention()

    dev = f"cuda:{a.gpu}"
    torch.cuda.set_device(dev)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16,
        attn_implementation="eager").to(dev).eval()

    ds = load_dataset("simonjegou/ruler", a.config, split="test")
    budgets = [float(x) for x in a.budgets.split(",")]
    rows = []
    with open(a.out, "w") as fh:
        for task in a.tasks.split(","):
            sub = ds.filter(lambda r: r["task"] == task).select(range(a.max_examples))
            for idx, ex in enumerate(sub):
                prompt = ge.build_prompt(tok, ex)
                ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
                T = ids.shape[1]
                with torch.no_grad():
                    out = model(input_ids=ids, output_attentions=True,
                                use_cache=True, return_dict=True)
                past = out.past_key_values
                keys = [l.keys.clone() for l in past.layers]
                vals = [l.values.clone() for l in past.layers]
                scores, needs = layer_scores_and_need(
                    out.attentions, a.obs_window, T, dev)
                drop, _per_layer = ge.compute_drop_from_attentions(
                    out.attentions, a.obs_window, a.top_k)
                del out, past
                torch.cuda.empty_cache()
                L = len(keys)

                for b in budgets:
                    budget = max(a.obs_window + a.n_sink + 4, int(T * b))
                    mid = max(0, budget - a.n_sink - a.obs_window) * L
                    res = {}
                    for arm in ("plain", "uniform"):
                        if b >= 1.0:
                            masks = torch.ones(L, T, dtype=torch.bool, device=dev)
                        elif arm == "plain":
                            masks = masks_from(scores,
                                               adaptive_budgets(needs, mid, L),
                                               T, a.obs_window, a.n_sink, dev)
                        else:
                            uni = torch.full((L,), mid // L, dtype=torch.long,
                                             device=dev)
                            masks = masks_from(scores, uni, T, a.obs_window,
                                               a.n_sink, dev)
                        res[arm + "_kept"] = float(masks.float().mean().item())
                        cache = DynamicCache()
                        for i, (k, v) in enumerate(zip(keys, vals)):
                            cache.update(k, v, i)
                        _STATE["active"], _STATE["mask"] = True, masks
                        txt = ge.generate_from_past(model, tok, cache, ids,
                                                    a.max_new, dev, T)
                        _STATE["active"], _STATE["mask"] = False, None
                        res[arm] = ge.is_correct(txt, ex["answer"], task=task)

                    if drop >= a.tau:
                        gated = res["plain"]
                    else:
                        cache = DynamicCache()
                        for i, (k, v) in enumerate(zip(keys, vals)):
                            cache.update(k, v, i)
                        txt = ge.generate_from_past(model, tok, cache, ids,
                                                    a.max_new, dev, T)
                        gated = ge.is_correct(txt, ex["answer"], task=task)

                    row = {"task": task, "id": idx, "budget": b, "T": T,
                           "drop": float(drop), "gate_open": bool(drop >= a.tau),
                           "correct_plain": bool(res["plain"]),
                           "correct_uniform": bool(res["uniform"]),
                           "correct_gated": bool(gated),
                           "kept_plain": res["plain_kept"],
                           "kept_uniform": res["uniform_kept"]}
                    fh.write(json.dumps(row) + "\n")
                    rows.append(row)
                fh.flush()
                del keys, vals
                torch.cuda.empty_cache()
                print(f"  [{task} {idx+1}/{a.max_examples}] drop={drop:+.4f}",
                      flush=True)
    print(f"wrote {len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
