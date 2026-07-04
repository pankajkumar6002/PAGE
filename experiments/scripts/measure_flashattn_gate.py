"""Deployment cost of PAGE's attention-materialization (two-pass) gate at scale.

The PAGE gate needs prefill attention weights (per-head, per-key) to compute the
head-agreement drop D. Fused FlashAttention/SDPA never materializes them, so the
paper's escape is a TWO-PASS mode:
  pass 1: SDPA/flash prefill (no attention weights) -> KV cache of length T
  pass 2: eager re-forward of the last w=32 queries against all T keys, with
          output_attentions=True, giving a [B, H, w, T] attention map per layer,
          from which the head-agreement drop is scored.

This script measures, for a given model at several context lengths T:
  1. two-pass GATE COST = eager pass-2 re-forward (ms) + head-agreement-drop
     computation (ms), N>=10 medians, torch.cuda.synchronize'd.
  2. peak GPU memory of the pass-2 scoring pass (the eager w x T attention
     materialization, all layers) vs the SDPA prefill peak.
  3. a model-free synthetic sweep of the drop computation across (L, H) so the
     O(L H^2 k) head-count scaling can be fit to an exponent.

Reuses compute_drop_from_attentions / head_agreement_layer from gated_eviction.py
(identical scoring code path used in the accuracy experiments).

Usage:
  .venv/bin/python experiments/scripts/measure_flashattn_gate.py \
     --model Qwen/Qwen2.5-32B-Instruct --gpu 3 \
     --contexts 4096,16384,32768 --out experiments/results/flashattn_gate_qwen32b.jsonl
"""
import argparse, json, os, statistics, subprocess, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gated_eviction import compute_drop_from_attentions  # noqa: E402

MiB = 2 ** 20


def sync_t(dev):
    torch.cuda.synchronize(dev)
    return time.perf_counter()


def gpu_snapshot():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.free",
             "--format=csv,noheader"], text=True)
        return out.strip().splitlines()
    except Exception as e:
        return [f"nvidia-smi failed: {e}"]


def switch_impl(model, impl):
    model.config._attn_implementation = impl
    for layer in model.model.layers:
        m = getattr(layer, "self_attn", None)
        if m is not None:
            try:
                m.config._attn_implementation = impl
            except Exception:
                pass


def med_iqr(xs):
    xs = sorted(xs)
    if not xs:
        return None, None, None
    m = statistics.median(xs)
    n = len(xs)
    q1 = xs[max(0, n // 4)]
    q3 = xs[min(n - 1, (3 * n) // 4)]
    return m, q1, q3


def measure_context(model, T, w, top_k, n_prefill, n_gate, warmup, dev, vocab):
    rec = {"T": T, "w": w, "top_k": top_k,
           "L": model.config.num_hidden_layers,
           "H": model.config.num_attention_heads,
           "KV": model.config.num_key_value_heads}
    ids = torch.randint(0, vocab, (1, T), device=dev)

    # ---- SDPA prefill (full T, no attentions): deployment-fair prefill peak ----
    switch_impl(model, "sdpa")
    prefill_ms = []
    peak_prefill = 0
    for i in range(warmup + n_prefill):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
        t0 = sync_t(dev)
        with torch.no_grad():
            out = model(input_ids=ids, output_attentions=False, use_cache=True, return_dict=True)
        t1 = sync_t(dev)
        if i >= warmup:
            prefill_ms.append((t1 - t0) * 1000.0)
            peak_prefill = max(peak_prefill, torch.cuda.max_memory_allocated(dev))
        del out
    torch.cuda.empty_cache()
    m, q1, q3 = med_iqr(prefill_ms)
    rec["sdpa_prefill_ms_median"] = m
    rec["sdpa_prefill_ms_iqr"] = [q1, q3]
    rec["sdpa_prefill_peak_MiB"] = peak_prefill / MiB

    # ---- Build KV of length T-w by prefilling first T-w (sdpa) ----
    with torch.no_grad():
        pre = model(input_ids=ids[:, :T - w], output_attentions=False,
                    use_cache=True, return_dict=True)
    past = pre.past_key_values
    del pre
    torch.cuda.empty_cache()

    # ---- Pass-2: eager re-forward of last w queries + drop computation ----
    last_ids = ids[:, -w:]
    pos = torch.arange(T - w, T, device=dev).unsqueeze(0)
    cpos = torch.arange(T - w, T, device=dev)
    switch_impl(model, "eager")

    reforward_ms, drop_ms, gate_ms = [], [], []
    peak_scoring = 0
    drops = []
    for i in range(warmup + n_gate):
        torch.cuda.reset_peak_memory_stats(dev)
        t0 = sync_t(dev)
        with torch.no_grad():
            sc = model(input_ids=last_ids, past_key_values=past, position_ids=pos,
                       cache_position=cpos, output_attentions=True, use_cache=False,
                       return_dict=True)
        t1 = sync_t(dev)  # reforward end (synced)
        drop, _ = compute_drop_from_attentions(sc.attentions, w, top_k)
        t2 = time.perf_counter()  # drop is CPU-side; no GPU work to sync
        if i >= warmup:
            reforward_ms.append((t1 - t0) * 1000.0)
            drop_ms.append((t2 - t1) * 1000.0)
            gate_ms.append((t2 - t0) * 1000.0)
            peak_scoring = max(peak_scoring, torch.cuda.max_memory_allocated(dev))
            drops.append(float(drop))
        del sc
    switch_impl(model, "sdpa")
    del past
    torch.cuda.empty_cache()

    rec["attn_shape"] = [1, rec["H"], w, T]
    for name, xs in (("reforward_ms", reforward_ms), ("drop_ms", drop_ms), ("gate_ms", gate_ms)):
        m, q1, q3 = med_iqr(xs)
        rec[name + "_median"] = m
        rec[name + "_iqr"] = [q1, q3]
    rec["scoring_peak_MiB"] = peak_scoring / MiB
    rec["drop_value"] = statistics.median(drops) if drops else None
    # analytic attention-materialization bytes (all layers, bf16)
    rec["attn_materialization_MiB"] = rec["L"] * rec["H"] * w * T * 2 / MiB
    return rec


def synthetic_drop_sweep(shapes, T, w, top_k, n, warmup, dev):
    """Model-free timing of compute_drop_from_attentions for various (L,H).

    Timing of the drop computation depends only on the attention *shape*
    (top-k set size is fixed at top_k), not on model weights, so this isolates
    the O(L H^2 k) head-count scaling exactly with the real scoring code."""
    rows = []
    for name, L, H in shapes:
        attns = tuple(torch.rand(1, H, w, T, device=dev, dtype=torch.bfloat16) for _ in range(L))
        ts = []
        for i in range(warmup + n):
            torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            compute_drop_from_attentions(attns, w, top_k)
            t1 = time.perf_counter()
            if i >= warmup:
                ts.append((t1 - t0) * 1000.0)
        m, q1, q3 = med_iqr(ts)
        rows.append({"synthetic": True, "name": name, "L": L, "H": H, "T": T,
                     "pairs": H * (H - 1) // 2, "LH2": L * H * H,
                     "drop_ms_median": m, "drop_ms_iqr": [q1, q3]})
        del attns
        torch.cuda.empty_cache()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct")
    p.add_argument("--contexts", default="4096,16384,32768")
    p.add_argument("--w", type=int, default=32)
    p.add_argument("--top_k", type=int, default=32)
    p.add_argument("--n_prefill", type=int, default=5)
    p.add_argument("--n_gate", type=int, default=15)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--gpu", type=int, default=3)
    p.add_argument("--synthetic_only", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    dev = f"cuda:{args.gpu}"
    torch.cuda.set_device(dev)
    contexts = [int(x) for x in args.contexts.split(",")]

    f = open(args.out, "w")
    meta = {"meta": True, "model": args.model, "contexts": contexts, "w": args.w,
            "top_k": args.top_k, "n_prefill": args.n_prefill, "n_gate": args.n_gate,
            "warmup": args.warmup, "gpu": args.gpu,
            "gpu_name": torch.cuda.get_device_name(dev),
            "gpu_snapshot_start": gpu_snapshot(), "torch": torch.__version__}
    f.write(json.dumps(meta) + "\n"); f.flush()

    print(f"loading {args.model}")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    print(f"loaded in {time.time()-t0:.0f}s; weights "
          f"{sum(p.numel()*p.element_size() for p in model.parameters())/2**30:.1f} GiB")
    vocab = model.config.vocab_size

    if not args.synthetic_only:
        for T in contexts:
            try:
                rec = measure_context(model, T, args.w, args.top_k,
                                      args.n_prefill, args.n_gate, args.warmup, dev, vocab)
                rec["meta"] = False
                f.write(json.dumps(rec) + "\n"); f.flush()
                print(f"[T={T}] prefill {rec['sdpa_prefill_ms_median']:.0f}ms peak "
                      f"{rec['sdpa_prefill_peak_MiB']:.0f}MiB | gate "
                      f"{rec['gate_ms_median']:.1f}ms (reforward "
                      f"{rec['reforward_ms_median']:.1f} + drop {rec['drop_ms_median']:.1f}) "
                      f"scoring peak {rec['scoring_peak_MiB']:.0f}MiB")
            except torch.cuda.OutOfMemoryError as e:
                torch.cuda.empty_cache()
                err = {"meta": False, "T": T, "oom": True, "error": str(e)[:200]}
                f.write(json.dumps(err) + "\n"); f.flush()
                print(f"[T={T}] OOM: {str(e)[:120]}")

    # synthetic drop sweep across (L,H) for the three measured model families + 70B
    shapes = [("Qwen2.5-1.5B", 28, 12), ("Mistral-7B", 32, 32),
              ("Qwen2.5-32B", 64, 40), ("Llama-70B-class", 80, 64)]
    for T in contexts:
        rows = synthetic_drop_sweep(shapes, T, args.w, args.top_k,
                                    max(args.n_gate, 15), args.warmup, dev)
        for r in rows:
            f.write(json.dumps(r) + "\n"); f.flush()
        print(f"[synthetic T={T}] " +
              ", ".join(f"{r['name']}(H={r['H']}):{r['drop_ms_median']:.1f}ms" for r in rows))

    f.write(json.dumps({"meta": True, "gpu_snapshot_end": gpu_snapshot()}) + "\n")
    f.close()
    print("done ->", args.out)


if __name__ == "__main__":
    main()
