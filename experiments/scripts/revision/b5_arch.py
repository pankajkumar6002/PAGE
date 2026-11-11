#!/usr/bin/env python
"""PAGE on hybrid and MLA models (Table 44), with D and SnapKV eviction on the layers that keep a key-value cache. Exit 4 if a start-up check fails, exit 5 if --out already has rows."""
import argparse
import copy
import json
import math
import os
import sys
import time

import torch

SRC = os.environ.get("PAGE_SRC", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SRC)
from gated_eviction import derive_keep_mask, head_agreement_layer, is_correct  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import AttentionInterface, AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig  # noqa: E402
from transformers.integrations.sdpa_attention import sdpa_attention_forward  # noqa: E402

NEAR_TIE = 0.25


class CheckFailed(Exception):
    pass


CAP = {"on": False, "w": 32, "top_k": 32, "dev": None, "agree": {}, "score": {}}


def _record(module, window_probs):
    idx = module.layer_idx
    CAP["agree"][idx] = head_agreement_layer(window_probs, CAP["top_k"])
    CAP["score"][idx] = window_probs.float().mean(dim=(0, 1)).to(CAP["dev"])


def capturing_sdpa(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs):
    if CAP["on"] and query.shape[2] > 1 and not kwargs.get("sliding_window"):
        w = CAP["w"]
        B, H, Tq, Dh = query.shape
        Hk, Tk = key.shape[1], key.shape[2]
        if not (B == 1 and Tq == Tk):
            raise CheckFailed(f"capture expects a batch-1 prefill without past, got q {tuple(query.shape)} k {tuple(key.shape)}")
        s = scaling if scaling is not None else Dh ** -0.5
        q = query[0, :, -w:, :].reshape(Hk, H // Hk, w, Dh)
        k = key[0].unsqueeze(1)
        scores = (torch.matmul(q, k.transpose(-1, -2)) * s).reshape(H, w, Tk)
        i = torch.arange(w, device=scores.device).unsqueeze(1)
        j = torch.arange(Tk, device=scores.device).unsqueeze(0)
        scores = scores.masked_fill((j > (Tk - w + i)).unsqueeze(0), torch.finfo(scores.dtype).min)
        _record(module, torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype))
        del q, k, scores
    return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling,
                                  is_causal=is_causal, **kwargs)


def patch_eager(model):
    patched = set()
    for mod in model.modules():
        if hasattr(mod, "layer_idx") and hasattr(mod, "scaling"):
            pymod = sys.modules[type(mod).__module__]
            if pymod.__name__ in patched or not hasattr(pymod, "eager_attention_forward"):
                continue
            orig = pymod.eager_attention_forward

            def wrapped(module, query, key, value, attention_mask, *a, _orig=orig, **kw):
                out, weights = _orig(module, query, key, value, attention_mask, *a, **kw)
                if CAP["on"] and query.shape[2] > 1 and not kw.get("sliding_window"):
                    _record(module, weights[0, :, -CAP["w"]:, :])
                return out, weights

            pymod.eager_attention_forward = wrapped
            patched.add(pymod.__name__)
    return sorted(patched)


def get_kv(past, i):
    if hasattr(past, "layers"):
        return past.layers[i].keys, past.layers[i].values
    return past.key_cache[i], past.value_cache[i]


def set_kv(past, i, k, v):
    if hasattr(past, "layers"):
        past.layers[i].keys, past.layers[i].values = k, v
    else:
        past.key_cache[i], past.value_cache[i] = k, v


def sliced_copy(past, attn_idx, keep):
    saved = {i: get_kv(past, i) for i in attn_idx}
    for i in attn_idx:
        set_kv(past, i, None, None)
    try:
        c = copy.deepcopy(past)
    finally:
        for i in attn_idx:
            set_kv(past, i, *saved[i])
    for i in attn_idx:
        k, v = saved[i]
        if keep is None:
            set_kv(c, i, k, v)
        else:
            idx = keep.to(k.device)
            set_kv(c, i, k.index_select(2, idx), v.index_select(2, idx))
    return c


def build(tok, ex):
    user = ex["context"] + "\n\n" + ex["question"]
    if ex.get("answer_prefix"):
        user += "\n" + ex["answer_prefix"]
    msgs = [{"role": "system", "content": "Answer the question concisely."}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def token_id(tok, s):
    i = tok.convert_tokens_to_ids(s)
    return i if isinstance(i, int) and i >= 0 and i != tok.unk_token_id else None


def end_ids(model, tok, rule):
    ids = set()
    e = model.generation_config.eos_token_id
    ids.update(e if isinstance(e, (list, tuple)) else ([e] if e is not None else []))
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    im_end = token_id(tok, "<|im_end|>")
    if im_end is not None:
        ids.add(im_end)
    return sorted(ids)


def step(model, cache, tok_id, pos, cp, dev, attention_mask=None):
    with torch.no_grad():
        out = model(input_ids=tok_id, past_key_values=cache, position_ids=torch.tensor([[pos]], device=dev),
                    cache_position=torch.tensor([cp], device=dev), attention_mask=attention_mask, use_cache=True,
                    return_dict=True)
    return out.logits[:, -1, :].float(), out.past_key_values


def decode(model, tok, cache, first_id, T, max_new, dev, n_kept, eos, return_ids=False):
    out_ids, nxt, pos, cp = [], first_id, T - 1, n_kept
    for _ in range(max_new):
        logits, cache = step(model, cache, nxt, pos, cp, dev)
        nid = int(logits.argmax(dim=-1).item())
        out_ids.append(nid)
        if nid in eos:
            break
        nxt = torch.tensor([[nid]], device=dev)
        pos += 1
        cp += 1
    text = tok.decode(out_ids, skip_special_tokens=True).strip()
    return (text, out_ids) if return_ids else text


def greedy_reference(model, ids, max_new, eos, pad):
    saved = model.generation_config
    model.generation_config = GenerationConfig()
    try:
        gc = GenerationConfig(max_new_tokens=max_new, do_sample=False, num_beams=1, eos_token_id=sorted(eos), pad_token_id=pad,
                              output_scores=True, return_dict_in_generate=True, use_cache=True)
        with torch.no_grad():
            g = model.generate(ids, attention_mask=torch.ones_like(ids), generation_config=gc)
        return g.sequences[0, ids.shape[1]:].tolist(), [s[0].float() for s in g.scores]
    finally:
        model.generation_config = saved


def heldback_reference(model, ids, max_new, eos):
    toks, logits = [], []
    with torch.no_grad():
        cache = model(input_ids=ids[:, :-1], use_cache=True, logits_to_keep=1).past_key_values
        nxt = ids[:, -1:]
        for _ in range(max_new):
            o = model(input_ids=nxt, past_key_values=cache, use_cache=True)
            cache = o.past_key_values
            lg = o.logits[0, -1].float()
            t = int(lg.argmax())
            toks.append(t)
            logits.append(lg)
            if t in eos:
                break
            nxt = torch.tensor([[t]], device=ids.device)
    return toks, logits


def top2_gap(logits):
    v = logits.float().topk(2).values
    return float(v[0] - v[1])


def load(args):
    if args.device == "cpu":
        gpus, device_map, max_memory = [], {"": "cpu"}, None
    else:
        gpus = [int(g) for g in str(args.gpu).split(",")]
        torch.cuda.set_device(gpus[0])
        if len(gpus) == 1:
            device_map, max_memory = {"": f"cuda:{gpus[0]}"}, None
        else:
            mem = [int(m) for m in args.mem_gb.split(",")] if args.mem_gb else []
            if len(mem) != len(gpus):
                raise CheckFailed(f"--mem_gb needs one cap per card, got {args.mem_gb!r} for cards {gpus}")
            device_map, max_memory = "sequential", {g: f"{m}GiB" for g, m in zip(gpus, mem)}
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=False)
    impl = "sdpa" if args.attn == "sdpa" else "eager"
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, config=config, dtype=torch.bfloat16, attn_implementation=impl,
                                                 device_map=device_map, max_memory=max_memory, trust_remote_code=False).eval()
    placed = set(map(str, getattr(model, "hf_device_map", {}).values()))
    if args.device != "cpu" and placed & {"cpu", "disk", "meta"}:
        raise CheckFailed(f"part of the model was placed off the GPUs: {sorted(placed)}")
    return tok, model, gpus


def deepseek_v2_scaling(model):
    cfg = model.config.get_text_config(decoder=True)
    if cfg.model_type != "deepseek_v2":
        return [], 1.0
    rp = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {}
    msa, factor = rp.get("mscale_all_dim", 0), rp.get("factor", 1.0)
    if not msa or factor <= 1:
        return [], 1.0
    m = 0.1 * msa * math.log(factor) + 1.0
    mods = [mod for mod in model.modules() if type(mod).__name__ == "DeepseekV2Attention"]
    return mods, m * m


def mean_nll(model, ids, n=1024):
    n = min(n, ids.shape[1] - 1)
    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False, logits_to_keep=n + 1).logits[0, :-1].float()
    return float(torch.nn.functional.cross_entropy(logits, ids[0, -n:].to(logits.device)))


def peak(gpus):
    return ", ".join(f"cuda:{g} {torch.cuda.max_memory_allocated(g) / 2**30:.1f} GiB" for g in gpus) or "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--gpu", default="0", help="local card index, or a comma list for multi-card loading (main card first)")
    p.add_argument("--mem_gb", default="", help="per-card weight caps in GiB for multi-card loading, same order as --gpu")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="cpu is for smoke tests only")
    p.add_argument("--out", required=True)
    p.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    p.add_argument("--stop", default="union", choices=["union"])
    p.add_argument("--expect_kv", type=int, default=0)
    p.add_argument("--tasks", default="niah_multikey_3,vt,fwe,qa_1")
    p.add_argument("--n_per_task", type=int, default=50)
    p.add_argument("--budgets", default="1.0,0.5,0.25,0.125,0.0625")
    p.add_argument("--max_new", type=int, default=128)
    p.add_argument("--obs_window", type=int, default=32)
    p.add_argument("--n_sink", type=int, default=4)
    p.add_argument("--top_k", type=int, default=32)
    p.add_argument("--check_b_tol", type=float, default=NEAR_TIE,
                   help="check (b): a different first pick is accepted if max|dlogit| is at most this")
    p.add_argument("--check", action="store_true", help="accepted for compatibility; the checks always run")
    args = p.parse_args()

    if os.path.exists(args.out) and os.path.getsize(args.out) > 0:
        print(f"FAIL: {args.out} already has rows; set it aside before rerunning", flush=True)
        sys.exit(5)
    try:
        run(args)
    except CheckFailed as e:
        print(f"FAIL: {e}", flush=True)
        sys.exit(4)


def run(args):
    AttentionInterface.register("sdpa", capturing_sdpa)
    tok, model, gpus = load(args)
    dev = model.get_input_embeddings().weight.device
    CAP.update(w=args.obs_window, top_k=args.top_k, dev=dev)
    if args.attn == "eager":
        patched = patch_eager(model)
        print(f"eager functions wrapped in: {patched}", flush=True)
        if not patched:
            raise CheckFailed("no eager attention function found to wrap")
    eos = set(end_ids(model, tok, args.stop))
    pad = tok.pad_token_id if tok.pad_token_id is not None else min(eos)
    cfg = model.config.get_text_config(decoder=True)
    types = getattr(cfg, "layer_types", None) or getattr(cfg, "layers_block_type", None)
    print(f"model {args.model} class {type(model).__name__}, layers {cfg.num_hidden_layers}, attn={args.attn}, stop={args.stop}, "
          f"cards {gpus}, end tokens {sorted(eos)}, layer types {dict((t, list(types).count(t)) for t in set(types)) if types else None}, "
          f"device map {sorted(set(map(str, getattr(model, 'hf_device_map', {}).values())))}", flush=True)
    print(f"model generation config (not used by the reference): {model.generation_config.to_diff_dict()}", flush=True)

    budgets = [float(b) for b in args.budgets.split(",")]
    w = args.obs_window
    print(f"check (b) tolerance {args.check_b_tol}", flush=True)

    ds = load_dataset("simonjegou/ruler", "4096", split="test")
    tasks = args.tasks.split(",")

    ds_mods, ds_factor = deepseek_v2_scaling(model)
    if ds_mods:
        ex0 = ds.filter(lambda r: r["task"] == tasks[0])[0]
        ids0 = tok(build(tok, ex0), return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        native = [m.scaling for m in ds_mods]
        nll_native = mean_nll(model, ids0)
        for m, s in zip(ds_mods, native):
            m.scaling = s * ds_factor
        nll_fixed = mean_nll(model, ids0)
        print(f"DeepSeek-V2 YaRN softmax factor {ds_factor:.4f} on {len(ds_mods)} layers: scaling {native[0]:.5f} -> {ds_mods[0].scaling:.5f}; "
              f"loss on the first prompt's last 1,024 tokens {nll_native:.4f} native, {nll_fixed:.4f} with the factor", flush=True)
        if not nll_fixed < nll_native:
            raise CheckFailed("the YaRN softmax factor did not lower the loss; not applying it silently")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fout = open(args.out, "a")
    t0 = time.time()
    attn_idx_seen = None
    for task in tasks:
        sub = ds.filter(lambda r: r["task"] == task)
        sub = sub.select(range(min(args.n_per_task, len(sub))))
        for idx, ex in enumerate(sub):
            prompt = build(tok, ex)
            if idx == 0 and task == tasks[0]:
                print(f"prompt head {prompt[:120]!r}\nprompt tail {prompt[-160:]!r}", flush=True)
            ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
            T = ids.shape[1]
            CAP["agree"], CAP["score"] = {}, {}
            CAP["on"] = True
            try:
                with torch.no_grad():
                    out = model(input_ids=ids[:, :-1], use_cache=True, return_dict=True, logits_to_keep=1)
            finally:
                CAP["on"] = False
            past = out.past_key_values
            del out
            attn_idx = sorted(CAP["agree"])
            if attn_idx_seen is None:
                attn_idx_seen = attn_idx
                print(f"window attention captured on {len(attn_idx)} layers: {attn_idx}", flush=True)
                if not attn_idx:
                    raise CheckFailed("no attention layer was captured")
                if args.expect_kv and len(attn_idx) != args.expect_kv:
                    raise CheckFailed(f"check (d): {len(attn_idx)} layers hold a key-value cache, expected {args.expect_kv}")
                for i in attn_idx:
                    k, v = get_kv(past, i)
                    if k is None or k.shape[2] != T - 1:
                        raise CheckFailed(f"layer {i}: cache length {None if k is None else k.shape[2]}, expected {T - 1}")
                k0, v0 = get_kv(past, attn_idx[0])
                print(f"cache keys {tuple(k0.shape)} values {tuple(v0.shape)}", flush=True)
            elif attn_idx != attn_idx_seen:
                raise CheckFailed(f"attention layers changed between inputs: {attn_idx} vs {attn_idx_seen}")
            per_layer = [CAP["agree"][i] for i in attn_idx]
            third = max(1, len(per_layer) // 3)
            drop = sum(per_layer[:third]) / third - sum(per_layer[-third:]) / third
            score = torch.stack([CAP["score"][i] for i in attn_idx]).mean(dim=0)
            if args.device != "cpu":
                torch.cuda.empty_cache()
            Tc = T - 1
            first = ids[:, -1:]
            check_this = idx == 0
            for b in budgets:
                keep = derive_keep_mask(score, Tc, b, w, args.n_sink, dev).nonzero(as_tuple=True)[0]
                n_kept = int(keep.shape[0])
                if check_this and b in (0.25, 0.0625):
                    lg_slice, _ = step(model, sliced_copy(past, attn_idx, keep), first, T - 1, n_kept, dev)
                    mask = torch.zeros(1, T, dtype=torch.long, device=dev)
                    mask[0, keep] = 1
                    mask[0, T - 1] = 1
                    lg_ref, _ = step(model, sliced_copy(past, attn_idx, None), first, T - 1, Tc, dev, attention_mask=mask)
                    a1, a2 = int(lg_slice.argmax()), int(lg_ref.argmax())
                    md = float((lg_slice - lg_ref).abs().max())
                    print(f"  check (b) {task} b={b}: argmax {a1} vs {a2}, max|dlogit| {md:.4f}", flush=True)
                    if a1 != a2 and md > args.check_b_tol:
                        raise CheckFailed(f"check (b) {task} b={b}: sliced cache picks {a1}, masked full cache picks {a2}, max|dlogit| {md:.4f}")
                cache = sliced_copy(past, attn_idx, None if n_kept == Tc else keep)
                pred, pred_ids = decode(model, tok, cache, first, T, args.max_new, dev, n_kept, eos, return_ids=True)
                del cache
                ok = is_correct(pred, ex["answer"] if "answer" in ex else ex["answers"], task=task)
                rec = {"id": idx, "task": task, "budget": b, "T": T, "drop": drop, "gate_open": drop >= 0.07,
                       "n_kept_plain": n_kept, "pred_plain": pred[:300], "correct_plain": bool(ok)}
                if b == 1.0:
                    rec["per_layer"] = per_layer
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                if check_this and b == 1.0:
                    hb_ids, hb_logits = heldback_reference(model, ids, args.max_new, eos)
                    raw = tok.decode(hb_ids, skip_special_tokens=False)
                    if pred_ids == hb_ids:
                        print(f"  check (a) {task}: exact match with the standard held-back decode over {len(hb_ids)} tokens; ours={pred[:70]!r}", flush=True)
                    else:
                        k = next((i for i, (x, y) in enumerate(zip(pred_ids, hb_ids)) if x != y), min(len(pred_ids), len(hb_ids)))
                        gap = top2_gap(hb_logits[k]) if k < len(hb_logits) else float("inf")
                        print(f"  check (a) {task}: first difference from the standard held-back decode at token {k}, top-2 gap {gap:.4f}; "
                              f"ours={pred[:70]!r} ref={tok.decode(hb_ids, skip_special_tokens=True)[:70]!r}", flush=True)
                        if gap > NEAR_TIE:
                            raise CheckFailed(f"check (a) {task}: decode leaves the standard held-back decode at token {k}, top-2 gap {gap:.4f}")
                    if "<think>" in raw or "</think>" in raw:
                        raise CheckFailed(f"check (c) {task}: thinking trace in the full-cache output: {raw[:120]!r}")
                    ref_ids, scores = greedy_reference(model, ids, args.max_new, eos, pad)
                    d0 = float((scores[0] - hb_logits[0]).abs().max()) if scores and hb_logits else float("nan")
                    k2 = next((i for i, (x, y) in enumerate(zip(pred_ids, ref_ids)) if x != y), None)
                    print(f"  info {task}: full-prefill generate {'matches' if pred_ids == ref_ids else f'first differs at token {k2}'}; "
                          f"first-step max|dlogit| full prefill vs held back {d0:.4f}; generate={tok.decode(ref_ids, skip_special_tokens=True)[:60]!r}", flush=True)
            del past, score
            if args.device != "cpu":
                torch.cuda.empty_cache()
            if check_this:
                print(f"[check {task}] passed; peak memory {peak(gpus)}", flush=True)
            if (idx + 1) % 5 == 0:
                print(f"[{task} {idx + 1}] T={T} D={drop:.4f} {time.time() - t0:.0f}s", flush=True)
    fout.close()
    print(f"b5 done, peak memory {peak(gpus)}", flush=True)


if __name__ == "__main__":
    main()
