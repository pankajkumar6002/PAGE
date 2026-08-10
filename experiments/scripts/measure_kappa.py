"""T-6: measure the margin-to-noise ratio kappa that Theorem 2 assumes.

Theorem 2 bounds Pr[D >= tau] for capacity-bound inputs by
beta = 2 H^2 L T^2 exp(-kappa^2 / 4), which is only informative when beta < 1.
At the paper's own H=16, L=28, T=4096 that needs kappa > 10.24. Combined with
the informal margin cap m <= log T = 8.32 nats, it forces sigma_z <= 0.81,
a sub-unit pre-softmax logit-noise scale that the paper never measured.

This measures it. A measured refutation of one's own sufficient condition is a
better position than an unmeasured assumption, so the result is reported
whichever way it comes out.

Operationalisation, stated plainly because the surrogate does not fix it. The
theory defines m and sigma_z on a single-head model with a relevant set R and
a noise set N. Empirically, for each (layer, head, observation query) we take
the pre-softmax attention logits over the causally valid keys, treat the
top-k (k = 32, the same k the gate uses) as R and the remainder as N, and set

    m       = mean(logits in R) - mean(logits in N)
    sigma_z = std(logits in N)
    kappa   = m / sigma_z

This is the natural analogue but it is a choice, not a derivation: a different
split of R from N would move kappa. We therefore report the full distribution
rather than a point estimate, and the comparison against the threshold is
about order of magnitude, not the third decimal.
"""
import argparse
import json
import math
import os
import statistics

import torch


CAPTURE = []          # (layer_idx, pre-softmax logits) for the observation rows
_LAYER = {"i": 0}


def install_hook(obs_window):
    """Capture pre-softmax attention logits from the eager attention path."""
    from transformers.models.qwen2 import modeling_qwen2 as qwen2
    try:
        from transformers.models.llama import modeling_llama as llama
        mods = [qwen2, llama]
    except Exception:
        mods = [qwen2]
    try:
        from transformers.models.mistral import modeling_mistral as mistral
        mods.append(mistral)
    except Exception:
        pass

    for mod in mods:
        if not hasattr(mod, "eager_attention_forward"):
            continue
        orig = mod.eager_attention_forward

        def patched(module, query, key, value, attention_mask, scaling,
                    dropout=0.0, _orig=orig, **kw):
            # replicate the pre-softmax step rather than trusting kwargs
            from transformers.models.qwen2.modeling_qwen2 import repeat_kv
            n_rep = query.shape[1] // key.shape[1]
            k = repeat_kv(key, n_rep) if n_rep > 1 else key
            logits = torch.matmul(query, k.transpose(2, 3)) * scaling
            if attention_mask is not None:
                logits = logits + attention_mask[..., : k.shape[-2]]
            # keep only the last `obs_window` query rows; that is the window
            # the gate reads, and it bounds memory at O(w*T)
            CAPTURE.append(logits[:, :, -obs_window:, :].detach().float().cpu())
            return _orig(module, query, key, value, attention_mask, scaling,
                         dropout=dropout, **kw)

        mod.eager_attention_forward = patched
        if hasattr(mod, "ALL_ATTENTION_FUNCTIONS"):
            try:
                mod.ALL_ATTENTION_FUNCTIONS.register("eager", patched)
            except Exception:
                pass


def kappas_from_logits(logits, top_k):
    """kappa per (head, query row) for one layer's captured logits."""
    out = []
    _, H, W, T = logits.shape
    for h in range(H):
        for q in range(W):
            z = logits[0, h, q]
            # Causal masking adds finfo.min (~ -3.4e38), which is finite, so
            # isfinite() alone leaves masked keys in the noise set and blows up
            # both m and sigma_z. Drop anything in the mask's range.
            z = z[torch.isfinite(z) & (z > -1e30)]
            if z.numel() < top_k * 4:
                continue
            top = torch.topk(z, top_k).values
            thresh = top.min()
            noise = z[z < thresh]
            if noise.numel() < 8:
                continue
            sd = noise.std().item()
            if not math.isfinite(sd) or sd <= 0:
                continue
            out.append(((top.mean() - noise.mean()).item(), sd,
                        (top.mean() - noise.mean()).item() / sd))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--config", default="4096")
    ap.add_argument("--task", default="niah_multikey_3",
                    help="Theorem 2 concerns the capacity-bound class")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--obs_window", type=int, default=32)
    ap.add_argument("--top_k", type=int, default=32)
    ap.add_argument("--device_map", default=None,
                    help="'auto' shards across visible GPUs when one card OOMs")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    install_hook(a.obs_window)

    ds = load_dataset("simonjegou/ruler", a.config, split="test")
    ds = ds.filter(lambda r: r["task"] == a.task).select(range(a.n))
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation="eager",
        device_map=a.device_map or {"": 0})
    model.eval()

    rows = []
    for i, ex in enumerate(ds):
        CAPTURE.clear()
        ids = tok(ex["context"] + "\n" + ex["question"],
                  return_tensors="pt").input_ids
        ids = ids.to(next(model.parameters()).device)
        with torch.no_grad():
            model(input_ids=ids, use_cache=False)
        per_input = []
        for lyr, lg in enumerate(CAPTURE):
            for m, sd, k in kappas_from_logits(lg, a.top_k):
                per_input.append({"layer": lyr, "m": m, "sigma_z": sd, "kappa": k})
        ks = [r["kappa"] for r in per_input]
        rows.append({"id": i, "task": a.task, "T": int(ids.shape[1]),
                     "n_measured": len(ks),
                     "kappa_median": statistics.median(ks) if ks else None,
                     "kappa_p90": (sorted(ks)[int(0.9 * len(ks))] if ks else None),
                     "kappa_max": max(ks) if ks else None,
                     "m_median": statistics.median(r["m"] for r in per_input) if ks else None,
                     "sigma_z_median": statistics.median(r["sigma_z"] for r in per_input) if ks else None})
        print(f"  input {i}: kappa median={rows[-1]['kappa_median']:.2f} "
              f"p90={rows[-1]['kappa_p90']:.2f} max={rows[-1]['kappa_max']:.2f}",
              flush=True)
        CAPTURE.clear()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    med = statistics.median(r["kappa_median"] for r in rows)
    mx = max(r["kappa_max"] for r in rows)
    H = model.config.num_attention_heads
    L = model.config.num_hidden_layers
    T = statistics.median(r["T"] for r in rows)
    need = math.sqrt(4 * math.log(2 * H * H * L * T * T))
    print(f"\n=== {a.model}  {a.task}  (H={H}, L={L}, T~{T:.0f}) ===")
    print(f"measured kappa: median {med:.2f}, max over all heads {mx:.2f}")
    print(f"Theorem 2 needs kappa > {need:.2f} for beta < 1")
    print("VERDICT:", "bound is informative" if med > need else
          f"bound is VACUOUS at the measured margin (median kappa {med:.2f} "
          f"<< {need:.2f} required)")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
