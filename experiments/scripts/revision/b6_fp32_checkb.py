"""Check (b) of b5_arch.py in float32 on CPU (the GLM-4.7-Flash float32 check of Appendix I, Table 44).

Repeats the runner's check (b) with the runner's own functions (imported from b5_arch.py, which is not changed):
prefill the first T - 1 tokens with window capture on, derive the SnapKV keep set at b = 1/4 and 1/16, then compare
the first decoding step from the sliced cache with a full-cache step whose attention mask hides the evicted positions.
In float32 the two are equal up to summation order, so a slicing or position error cannot hide behind bf16 rounding.
Also reports the float32 first-step difference between a full prefill and the held-back path.

Pass: the same argmax and max|dlogit| <= 0.01 at both budgets. Exit 1 on failure.
Usage: b6_fp32_checkb.py --model M --task T [--idx 0] [--threads 32]
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch

RUNNER = str(Path(__file__).resolve().with_name("b5_arch.py"))
TOL = 0.01

spec = importlib.util.spec_from_file_location("b5", RUNNER)
b5 = importlib.util.module_from_spec(spec)
sys.modules["b5"] = b5
spec.loader.exec_module(b5)
from datasets import load_dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def top2(lg):
    v = lg.topk(2)
    return [int(i) for i in v.indices], float(v.values[0] - v.values[1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--idx", type=int, default=0)
    p.add_argument("--threads", type=int, default=32)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    dev = "cpu"
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="eager",
                                                 device_map={"": "cpu"}, trust_remote_code=False).eval()
    print(f"loaded {type(model).__name__} in {time.time() - t0:.0f}s, dtype {next(model.parameters()).dtype}", flush=True)
    b5.CAP.update(w=32, top_k=32, dev=dev)
    print("eager wrapped:", b5.patch_eager(model), flush=True)
    mods, _ = b5.deepseek_v2_scaling(model)
    if mods:
        sys.exit("this check is for models without the DeepSeek-V2 softmax factor")

    ds = load_dataset("simonjegou/ruler", "4096", split="test")
    ex = ds.filter(lambda r: r["task"] == args.task)[args.idx]
    ids = tok(b5.build(tok, ex), return_tensors="pt", add_special_tokens=False).input_ids
    T = ids.shape[1]
    b5.CAP["agree"], b5.CAP["score"] = {}, {}
    b5.CAP["on"] = True
    try:
        with torch.no_grad():
            out = model(input_ids=ids[:, :-1], use_cache=True, return_dict=True, logits_to_keep=1)
    finally:
        b5.CAP["on"] = False
    past = out.past_key_values
    del out
    attn_idx = sorted(b5.CAP["agree"])
    per_layer = [b5.CAP["agree"][i] for i in attn_idx]
    third = max(1, len(per_layer) // 3)
    drop = sum(per_layer[:third]) / third - sum(per_layer[-third:]) / third
    score = torch.stack([b5.CAP["score"][i] for i in attn_idx]).mean(dim=0)
    Tc, first = T - 1, ids[:, -1:]
    print(f"{args.model} {args.task}[{args.idx}] float32 T={T}, {len(attn_idx)} key-value layers, D={drop:.4f} ({time.time() - t0:.0f}s)",
          flush=True)

    ok = True
    for b in (0.25, 0.0625):
        keep = b5.derive_keep_mask(score, Tc, b, 32, 4, dev).nonzero(as_tuple=True)[0]
        n_kept = int(keep.shape[0])
        lg_slice, _ = b5.step(model, b5.sliced_copy(past, attn_idx, keep), first, T - 1, n_kept, dev)
        mask = torch.zeros(1, T, dtype=torch.long, device=dev)
        mask[0, keep] = 1
        mask[0, T - 1] = 1
        lg_ref, _ = b5.step(model, b5.sliced_copy(past, attn_idx, None), first, T - 1, Tc, dev, attention_mask=mask)
        (s1, s2), gs = top2(lg_slice[0])
        (r1, r2), gr = top2(lg_ref[0])
        md = float((lg_slice - lg_ref).abs().max())
        this_ok = s1 == r1 and md <= TOL
        ok &= this_ok
        print(f"check (b) float32 {args.task} b={b} n_kept={n_kept}: argmax sliced {s1} vs masked {r1}; max|dlogit| {md:.6f}; "
              f"top-2 sliced {s1},{s2} gap {gs:.4f}, masked {r1},{r2} gap {gr:.4f}; max|logit| {float(lg_ref.abs().max()):.1f}; "
              f"{'pass' if this_ok else 'FAIL'} ({time.time() - t0:.0f}s)", flush=True)

    with torch.no_grad():
        lf = model(input_ids=ids, logits_to_keep=1).logits[0, -1].float()
    lh, _ = b5.step(model, b5.sliced_copy(past, attn_idx, None), first, T - 1, Tc, dev)
    print(f"info float32 {args.task}: first step full prefill vs held back, argmax {int(lf.argmax())} vs {int(lh[0].argmax())}, "
          f"max|dlogit| {float((lf - lh[0]).abs().max()):.6f} ({time.time() - t0:.0f}s)", flush=True)
    print(f"RESULT {args.task}: {'pass' if ok else 'FAIL'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
