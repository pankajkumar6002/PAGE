"""End-to-end latency / memory measurement for gated KV-cache eviction.

Measures, per input (RULER 4K, batch size 1, bfloat16, greedy):
  - prefill time WITHOUT output_attentions (deployment-fair full-KV baseline)
  - prefill time WITH output_attentions (the one-pass scoring machinery)
  - SnapKV scoring time (pool_score over the retained attentions)
  - head-agreement-drop (gate) computation time
  - decode throughput over a fixed number of steps (no EOS early-exit, so
    every config amortizes over the same token count)
  - decode-phase peak GPU memory (reset after cache construction) and the
    actual KV-cache tensor bytes per config

Configs: full KV, plain SnapKV b=0.125, plain SnapKV b=0.0625, and gated
SnapKV (evict at b=0.125 iff head-agreement drop >= tau, else keep full KV;
the gate signal is always computed, so gated timings include gate overhead).

Each config re-runs its own prefill so that per-config peak memory is not
contaminated by tensors another config needed (e.g. retained attentions).

Usage:
  .venv/bin/python experiments/scripts/measure_latency.py \
    --gpu 1 --out experiments/results/latency_qwen15b_4k.jsonl
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gated_eviction import (  # noqa: E402
    build_prompt,
    compute_drop_from_attentions,
    derive_keep_mask,
    pool_score,
)


def sync_time(device):
    torch.cuda.synchronize(device)
    return time.perf_counter()


def gpu_snapshot(gpu_index):
    """Record background utilization/memory on the measurement GPU."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader"],
            text=True,
        )
        return out.strip().splitlines()
    except Exception as e:  # pragma: no cover
        return [f"nvidia-smi failed: {e}"]


def kv_bytes_of(past):
    total = 0
    for layer in past.layers:
        total += layer.keys.numel() * layer.keys.element_size()
        total += layer.values.numel() * layer.values.element_size()
    return total


def build_pruned_cache(past, keep_idx):
    pruned = DynamicCache()
    for i, layer in enumerate(past.layers):
        pruned.update(
            layer.keys.index_select(2, keep_idx),
            layer.values.index_select(2, keep_idx),
            i,
        )
    return pruned


def timed_decode(model, past, ids, n_steps, device, original_T):
    """Greedy decode for exactly n_steps (no EOS break), timed with syncs.

    Returns (elapsed_seconds, generated_ids). Position/cache bookkeeping is
    identical to gated_eviction.generate_from_past.
    """
    n_kept = past.layers[0].keys.shape[2]
    next_token = ids[:, -1:].clone()
    next_position = torch.tensor([[original_T - 1]], dtype=torch.long, device=device)
    cache_position = torch.tensor([n_kept], dtype=torch.long, device=device)

    out_ids = []
    t0 = sync_time(device)
    for _ in range(n_steps):
        with torch.no_grad():
            out = model(
                input_ids=next_token,
                past_key_values=past,
                position_ids=next_position,
                cache_position=cache_position,
                use_cache=True,
                return_dict=True,
            )
        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        out_ids.append(int(next_token.item()))
        next_position = next_position + 1
        cache_position = cache_position + 1
    t1 = sync_time(device)
    return t1 - t0, out_ids


def run_config(model, ids, cfg, args, device):
    """Run one config end-to-end on one input. Returns a metrics dict.

    cfg in {"full", "b0.125", "b0.0625", "gated"}.
    """
    T = ids.shape[1]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    m = {"config": cfg, "T": T}

    need_attn = cfg != "full"
    t0 = sync_time(device)
    with torch.no_grad():
        out = model(
            input_ids=ids,
            output_attentions=need_attn,
            use_cache=True,
            return_dict=True,
        )
    t1 = sync_time(device)
    m["t_prefill_ms"] = (t1 - t0) * 1000.0
    past = out.past_key_values

    drop = None
    if need_attn:
        t0 = sync_time(device)
        score = pool_score(out.attentions, "snapkv", args.obs_window, T, device)
        t1 = sync_time(device)
        m["t_snapkv_score_ms"] = (t1 - t0) * 1000.0

        t0 = sync_time(device)
        drop, _ = compute_drop_from_attentions(out.attentions, args.obs_window, args.top_k)
        t1 = sync_time(device)
        m["t_gate_drop_ms"] = (t1 - t0) * 1000.0
        m["drop"] = float(drop)

    # Decide keep set.
    if cfg == "full":
        do_evict, budget = False, 1.0
    elif cfg == "gated":
        do_evict, budget = (drop >= args.tau), args.gated_budget
        m["gate_open"] = bool(do_evict)
    else:
        do_evict, budget = True, float(cfg[1:])

    if do_evict and budget < 1.0:
        t0 = sync_time(device)
        keep_mask = derive_keep_mask(score, T, budget, args.obs_window, args.n_sink,
                                     device, policy="snapkv")
        keep_idx = keep_mask.nonzero(as_tuple=True)[0]
        cache = build_pruned_cache(past, keep_idx)
        t1 = sync_time(device)
        m["t_evict_ms"] = (t1 - t0) * 1000.0
        m["n_kept"] = int(keep_idx.shape[0])
    else:
        cache = past  # decode directly from the prefill cache
        m["n_kept"] = T

    m["kv_bytes"] = kv_bytes_of(cache)
    m["peak_prefill_bytes"] = torch.cuda.max_memory_allocated(device)

    # Free everything the decode phase does not need, then measure decode peak.
    attn_alive = out.attentions if need_attn else None
    del out
    if cache is not past:
        del past
    del attn_alive
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    elapsed, out_ids = timed_decode(model, cache, ids, args.max_new, device, T)
    m["t_decode_s"] = elapsed
    m["decode_tok_s"] = args.max_new / elapsed
    m["peak_decode_bytes"] = torch.cuda.max_memory_allocated(device)
    m["n_decoded"] = len(out_ids)

    del cache
    torch.cuda.empty_cache()
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--config", default="4096")
    p.add_argument("--tasks", default="niah_multikey_3,vt,fwe,qa_1")
    p.add_argument("--n_per_task", type=int, default=5)
    p.add_argument("--n_warmup", type=int, default=3)
    p.add_argument("--max_new", type=int, default=128)
    p.add_argument("--obs_window", type=int, default=32)
    p.add_argument("--n_sink", type=int, default=4)
    p.add_argument("--top_k", type=int, default=32)
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--gated_budget", type=float, default=0.125)
    p.add_argument("--configs", default="full,b0.125,b0.0625,gated")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(device)
    configs = args.configs.split(",")
    tasks = args.tasks.split(",")

    print("GPU snapshot at start:")
    start_snapshot = gpu_snapshot(args.gpu)
    for line in start_snapshot:
        print(" ", line)

    print(f"loading RULER {args.config}")
    ds = load_dataset("simonjegou/ruler", args.config, split="test")
    measured_rows, warmup_rows = [], []
    for task in tasks:
        sub = ds.filter(lambda r: r["task"] == task)
        n = min(args.n_per_task, len(sub))
        for r in sub.select(range(n)):
            measured_rows.append(r)
        # warmup rows come AFTER the measured indices so they never overlap
        if len(warmup_rows) < args.n_warmup and len(sub) > n:
            warmup_rows.append(sub[n])
    warmup_rows = warmup_rows[: args.n_warmup]
    print(f"{len(measured_rows)} measured rows ({args.n_per_task}/task), "
          f"{len(warmup_rows)} warmup rows")

    print(f"loading {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
    ).to(device).eval()
    weights_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"model weights: {weights_bytes/2**20:.0f} MiB")

    results_f = open(args.out, "w")
    meta = {
        "meta": True,
        "model": args.model,
        "ruler_config": args.config,
        "tasks": tasks,
        "configs": configs,
        "tau": args.tau,
        "gated_budget": args.gated_budget,
        "max_new": args.max_new,
        "gpu": args.gpu,
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_snapshot_start": start_snapshot,
        "weights_bytes": weights_bytes,
        "torch": torch.__version__,
        "attn_implementation": "eager",
    }
    results_f.write(json.dumps(meta) + "\n")
    results_f.flush()

    t_start = time.time()
    for phase, rows in (("warmup", warmup_rows), ("measure", measured_rows)):
        for ex_i, ex in enumerate(rows):
            prompt = build_prompt(tokenizer, ex)
            ids = tokenizer(prompt, return_tensors="pt",
                            add_special_tokens=False).input_ids.to(device)
            for cfg in configs:
                m = run_config(model, ids, cfg, args, device)
                m.update({"phase": phase, "id": ex_i, "task": ex["task"]})
                results_f.write(json.dumps(m) + "\n")
                results_f.flush()
            print(f"[{phase} {ex_i+1}/{len(rows)}] task={ex['task']} T={ids.shape[1]} "
                  f"elapsed={time.time()-t_start:.0f}s")

    end_snapshot = gpu_snapshot(args.gpu)
    results_f.write(json.dumps({"meta": True, "gpu_snapshot_end": end_snapshot}) + "\n")
    results_f.close()
    print("GPU snapshot at end:")
    for line in end_snapshot:
        print(" ", line)
    print(f"done in {time.time()-t_start:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
