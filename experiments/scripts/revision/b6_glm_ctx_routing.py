"""GLM-4.7-Flash, float32 on CPU, standard model code: do context tokens get different experts in a prefill of all T
tokens than in a prefill of the first T - 1 tokens? In exact arithmetic the first T - 1 tokens are identical in both.
A routing flip on a context token changes its keys and values in later layers, which the last token's attention then
reads. That would explain a float32 full-prefill vs held-back gap that float64 removes (3e-13).
Prints, per MoE layer, how many context tokens chose a different expert set, and the first layer with a flip.
Usage: b6_glm_ctx_routing.py --model M --task T [--idx 0] [--threads 32]
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location("b5", str(Path(__file__).resolve().with_name("b5_arch.py")))
b5 = importlib.util.module_from_spec(spec)
sys.modules["b5"] = b5
spec.loader.exec_module(b5)
from datasets import load_dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

REC = {"on": False, "idx": {}, "out": {}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--idx", type=int, default=0)
    p.add_argument("--threads", type=int, default=32)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="eager",
                                                 device_map={"": "cpu"}, trust_remote_code=False).eval()
    layers = model.model.layers
    for i, layer in enumerate(layers):
        layer.register_forward_hook(lambda m, a, o, i=i: REC["out"].__setitem__(i, (o[0] if isinstance(o, tuple) else o)[0].detach().clone())
                                    if REC["on"] else None)
        moe = layer.mlp
        if hasattr(moe, "route_tokens_to_experts"):
            orig = moe.route_tokens_to_experts

            def wrapped(router_logits, _orig=orig, i=i):
                idx, w = _orig(router_logits)
                if REC["on"]:
                    REC["idx"][i] = idx.sort(dim=-1).values.clone()
                return idx, w

            moe.route_tokens_to_experts = wrapped
    print(f"loaded in {time.time() - t0:.0f}s; experts implementation {getattr(model.config, '_experts_implementation', None)}", flush=True)
    ds = load_dataset("simonjegou/ruler", "4096", split="test")
    ex = ds.filter(lambda r: r["task"] == args.task)[args.idx]
    ids = tok(b5.build(tok, ex), return_tensors="pt", add_special_tokens=False).input_ids
    T = ids.shape[1]
    with torch.no_grad():
        REC["on"] = True
        model(input_ids=ids, logits_to_keep=1)
        A = {k: dict(v) for k, v in REC.items() if k != "on"}
        REC["idx"], REC["out"] = {}, {}
        model(input_ids=ids[:, :-1], logits_to_keep=1)
        REC["on"] = False
        B = {k: dict(v) for k, v in REC.items() if k != "on"}
    n = T - 1
    first = None
    print(f"{args.task}[{args.idx}] float32 T={T}: context tokens 0..{n - 1} ({time.time() - t0:.0f}s)", flush=True)
    for i in range(len(layers)):
        ho = A["out"][i][:n]
        dh = float(((ho - B["out"][i][:n]).norm(dim=-1) / ho.norm(dim=-1)).max())
        if i in A["idx"]:
            diff = (A["idx"][i][:n] != B["idx"][i][:n]).any(dim=-1).nonzero().flatten().tolist()
            if diff and first is None:
                first = i
            print(f"layer {i:2d}: context tokens with a different expert set {len(diff):4d}"
                  f"{'' if not diff else f' (first positions {diff[:5]})'}; max rel|d out| over context {dh:.2e}", flush=True)
        else:
            print(f"layer {i:2d}: dense; max rel|d out| over context {dh:.2e}", flush=True)
    print(f"first layer with a context routing flip: {first}", flush=True)


if __name__ == "__main__":
    main()
