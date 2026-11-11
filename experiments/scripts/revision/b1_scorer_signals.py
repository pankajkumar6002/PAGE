#!/usr/bin/env python
"""Per-input scorer signals from one prefill (Table 41)."""
import argparse
import json
import math
import os
import sys
import time

import torch

SRC = os.environ.get("PAGE_SRC", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SRC)
from gated_eviction import build_prompt, derive_keep_mask, head_agreement_layer  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402

SCORERS = ["snapkv", "h2o", "obcache_v", "obcache_k", "obcache_vk"]


def rotate_half(x):
    n = x.shape[-1] // 2
    return torch.cat((-x[..., n:], x[..., :n]), dim=-1)


class Capture:
    def __init__(self, layers):
        self.enabled = True
        self.store = {}
        self.handles = []
        for i, layer in enumerate(layers):
            self.handles.append(layer.self_attn.register_forward_pre_hook(self._hook(i), with_kwargs=True))

    def _hook(self, i):
        def fn(module, args, kwargs):
            if not self.enabled:
                return
            hs = kwargs.get("hidden_states", args[0] if args else None)
            self.store[i] = (hs.detach(), kwargs["position_embeddings"])
        return fn


def query_states(module, hs, pe, start, end):
    H = module.config.num_attention_heads
    d = module.head_dim
    q = module.q_proj(hs[:, start:end])
    q = q.view(1, end - start, H, d).transpose(1, 2)
    cos, sin = pe
    cos, sin = cos[:, start:end], sin[:, start:end]
    return q * cos.unsqueeze(1) + rotate_half(q) * sin.unsqueeze(1)


def window_attention(module, hs, pe, keys, start, end):
    T = keys.shape[2]
    H = module.config.num_attention_heads
    groups = H // keys.shape[1]
    q = query_states(module, hs, pe, start, end)
    k = keys.repeat_interleave(groups, dim=1)
    z = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(module.head_dim)
    qp = torch.arange(start, end, device=keys.device)
    kp = torch.arange(T, device=keys.device)
    mask = kp.unsqueeze(0) > qp.unsqueeze(1)
    z = z.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    a = torch.nn.functional.softmax(z, dim=-1, dtype=torch.float32).to(q.dtype)
    return a, z


def obcache_scores(a, z, values, w):
    H_kv = values.shape[1]
    groups = a.shape[1] // H_kv
    A = a.float().view(1, H_kv, groups, *a.shape[2:]).mean(dim=2)
    Z = z.float().view(1, H_kv, groups, *z.shape[2:]).mean(dim=2)
    V = values.float()
    O = torch.matmul(A, V)
    A, Z, V = A[..., :-w], Z[..., :-w], V[:, :, :-w, :]
    vnorm = V.pow(2).sum(dim=-1)
    onorm = O.pow(2).sum(dim=-1)
    dot = torch.einsum("bhqd,bhpd->bhqp", O, V)
    s_v = A.pow(2).sum(dim=-2) * vnorm
    diff = vnorm[..., None, :] + onorm[..., None] - 2.0 * dot
    s_k = ((A * Z).pow(2) * diff).sum(dim=-2)
    cross = (A.pow(2) * Z * (vnorm[..., None, :] - dot)).sum(dim=-2)
    s_vk = s_v + s_k + 2.0 * cross
    return s_v[0], s_k[0], s_vk[0]


def layer_normalized(per_layer):
    out = []
    for s in per_layer:
        out.append(s / (s.abs().sum() + 1e-12))
    return torch.stack(out).mean(dim=0)


def middle_share(score, T, b, w, n_sink, device):
    keep = derive_keep_mask(score, T, b, w, n_sink, device)
    mid = torch.zeros(T, dtype=torch.bool, device=device)
    mid[n_sink:T - w] = True
    s = score.clamp(min=0)
    total = s[mid].sum()
    return float(s[mid & keep].sum() / total) if total > 0 else float("nan")


def first_step(model, ids, keys, values, T, keep_idx, device):
    past = DynamicCache()
    for i, (k, v) in enumerate(zip(keys, values)):
        past.update(k.index_select(2, keep_idx), v.index_select(2, keep_idx), i)
    n = int(keep_idx.shape[0])
    with torch.no_grad():
        out = model(
            input_ids=ids[:, -1:],
            past_key_values=past,
            position_ids=torch.tensor([[T - 1]], dtype=torch.long, device=device),
            cache_position=torch.tensor([n], dtype=torch.long, device=device),
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
    h = out.hidden_states[-1][0, -1].float().clone()
    logits = out.logits[0, -1].float().clone()
    del out, past
    return h, logits


def load_logged(path):
    by = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if float(r["budget"]) == 1.0:
                by.setdefault(r["task"], []).append((r["id"], r["T"], r["drop"]))
    return {t: [(T, d) for _, T, d in sorted(v)] for t, v in by.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--logged", required=True, help="gated log used for validation")
    p.add_argument("--tasks", default="niah_multikey_3,vt,fwe,qa_1")
    p.add_argument("--n_per_task", type=int, default=100)
    p.add_argument("--config", default="4096")
    p.add_argument("--budgets", default="0.0625,0.125,0.25,0.5")
    p.add_argument("--obs_window", type=int, default=32)
    p.add_argument("--n_sink", type=int, default=4)
    p.add_argument("--top_k", type=int, default=32)
    p.add_argument("--h2o_chunk", type=int, default=256)
    p.add_argument("--validate_first", type=int, default=5)
    p.add_argument("--drop_tol", type=float, default=0.005)
    args = p.parse_args()

    budgets = [float(b) for b in args.budgets.split(",")]
    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)
    w = args.obs_window
    logged = load_logged(args.logged)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="eager").to(device).eval()
    layers = model.model.layers
    cap = Capture(layers)
    L = len(layers)
    third = max(1, L // 3)

    ds = load_dataset("simonjegou/ruler", args.config, split="test")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fout = open(args.out, "a")
    t0 = time.time()
    done = 0
    for task in args.tasks.split(","):
        sub = ds.filter(lambda r: r["task"] == task)
        sub = sub.select(range(min(args.n_per_task, len(sub))))
        for idx, ex in enumerate(sub):
            prompt = build_prompt(tok, ex)
            ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            T = ids.shape[1]
            cap.enabled = True
            cap.store = {}
            with torch.no_grad():
                out = model(input_ids=ids, output_attentions=False, use_cache=True, return_dict=True)
            cap.enabled = False
            past = out.past_key_values
            keys = [x.keys.clone() for x in past.layers]
            values = [x.values.clone() for x in past.layers]
            del out, past

            per_agree, ents, snap, h2o, s_v, s_k, s_vk = [], [], [], [], [], [], []
            with torch.no_grad():
                for i, layer in enumerate(layers):
                    hs, pe = cap.store[i]
                    mod = layer.self_attn
                    a, z = window_attention(mod, hs, pe, keys[i], T - w, T)
                    per_agree.append(head_agreement_layer(a[0], args.top_k))
                    pr = a[0].float().clamp(min=1e-12)
                    ents.append(float((-(pr * pr.log()).sum(dim=-1)).mean() / math.log(max(T, 2))))
                    snap.append(a[0].float().mean(dim=(0, 1)))
                    v_, k_, vk_ = obcache_scores(a, z, values[i], w)
                    s_v.append(v_.mean(dim=0))
                    s_k.append(k_.mean(dim=0))
                    s_vk.append(vk_.mean(dim=0))
                    acc = None
                    for st in range(0, T, args.h2o_chunk):
                        aa, _ = window_attention(mod, hs, pe, keys[i], st, min(T, st + args.h2o_chunk))
                        part = aa[0].float().sum(dim=1)
                        acc = part if acc is None else acc + part
                        del aa
                    h2o.append((acc / T).mean(dim=0))
                    del a, z, acc
            cap.store = {}
            drop = sum(per_agree[:third]) / third - sum(per_agree[L - third:]) / third
            a_early, a_late = sum(per_agree[:third]) / third, sum(per_agree[L - third:]) / third
            ent_drop = sum(ents[:third]) / third - sum(ents[L - third:]) / third

            lg = logged.get(task, [])
            if idx < args.validate_first:
                if idx >= len(lg) or int(lg[idx][0]) != int(T) or abs(lg[idx][1] - drop) > args.drop_tol:
                    print(f"VALIDATION FAILED {task}/{idx}: T={T} D={drop:.5f} logged={lg[idx] if idx < len(lg) else None}", flush=True)
                    sys.exit(3)
                print(f"validated {task}/{idx}: T={T} D={drop:.5f} logged D={lg[idx][1]:.5f}", flush=True)

            pad = torch.zeros(w, device=device)
            scores = {
                "snapkv": torch.stack(snap).mean(dim=0),
                "h2o": torch.stack(h2o).mean(dim=0),
                "obcache_v": torch.cat([layer_normalized(s_v), pad]),
                "obcache_k": torch.cat([layer_normalized(s_k), pad]),
                "obcache_vk": torch.cat([layer_normalized(s_vk), pad]),
            }
            shares = {n: {str(b): middle_share(s.to(device), T, b, w, args.n_sink, device) for b in budgets} for n, s in scores.items()}
            sn = scores["snapkv"]
            topk_share = float(sn.topk(min(args.top_k, T)).values.sum() / sn.sum())

            h_full, lg_full = first_step(model, ids, keys, values, T, torch.arange(T, device=device), device)
            p_full = torch.log_softmax(lg_full, dim=-1)
            act, kl, same = {}, {}, {}
            for b in budgets:
                mask = derive_keep_mask(sn, T, b, w, args.n_sink, device)
                h_b, lg_b = first_step(model, ids, keys, values, T, mask.nonzero(as_tuple=True)[0], device)
                act[str(b)] = float((h_b - h_full).norm() / (h_full.norm() + 1e-12))
                kl[str(b)] = float((p_full.exp() * (p_full - torch.log_softmax(lg_b, dim=-1))).sum())
                same[str(b)] = bool(lg_b.argmax() == lg_full.argmax())

            rec = {"task": task, "id": idx, "T": T, "drop": drop, "per_layer": per_agree, "a_early": a_early, "a_late": a_late,
                   "entropy_drop": ent_drop, "topk_share": topk_share, "shares": shares, "act_change": act, "kl": kl, "top1_same": same}
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            del keys, values, scores, snap, h2o, s_v, s_k, s_vk
            torch.cuda.empty_cache()
            done += 1
            if done % 10 == 0:
                print(f"[{done}] {task}/{idx} T={T} {time.time() - t0:.0f}s", flush=True)
    fout.close()
    print("b1 done", flush=True)


if __name__ == "__main__":
    main()
