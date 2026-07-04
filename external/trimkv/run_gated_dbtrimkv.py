"""Gated DBTrimKV vs plain DBTrimKV head-to-head on RULER mixed suite.

Two policies per example x budget:
  - plain   : always run DBTrimKV compress() at memory_size M
  - gated   : if drop >= tau, run DBTrimKV compress() at M; else skip
              compress entirely (full paged KV).

The gate signal is the early-vs-late layer head-agreement drop computed on
the prefill attentions of the BASE Qwen3-4B-Instruct-2507 model (loaded
separately in eager attention).  We mirror the same algorithm as
experiments/scripts/gated_eviction.py:head_agreement_layer and
compute_drop_from_attentions.  DBTrimKV inference itself uses its native
flash-attention 2 + paged cache path.

The cache-level integration handle is PagedTrimKVCache.do_compress.  When
False, the compress() method (called at the end of every model forward in
TrimKVQwen3Model) returns immediately, leaving the paged cache to grow
unbounded.  When True (default when memory_size is set), compress() fires
on prefill and emits the usual DBTrimKV eviction.

Output JSONL schema (one record per example x budget):
  id, task, budget, T, drop, gate_open,
  gold, pred_plain, pred_gated, correct_plain, correct_gated,
  n_kept_plain, n_kept_gated.
"""
import argparse
import gc
import itertools
import json
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  # GPU 1 free per nvidia-smi
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
torch.set_grad_enabled(False)

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from trimkv.models.qwen3 import TrimKVQwen3ForCausalLM
from trimkv.cache_utils import PagedTrimKVCache


# ---------------- Gate signal (mirrors gated_eviction.py) ----------------

def _jaccard(s1, s2):
    inter = len(s1 & s2)
    union = len(s1 | s2)
    return inter / union if union else 0.0


def head_agreement_layer(attn_layer, top_k):
    """attn_layer: [H, W, T]. Mean pairwise Jaccard of top-k attended keys."""
    H, W, T = attn_layer.shape
    head_mean = attn_layer.mean(dim=1)  # [H, T]
    top_idx = head_mean.topk(min(top_k, T), dim=-1).indices
    sets = [set(top_idx[h].cpu().tolist()) for h in range(H)]
    pairs = list(itertools.combinations(range(H), 2))
    if not pairs:
        return 0.0
    return sum(_jaccard(sets[i], sets[j]) for i, j in pairs) / len(pairs)


def compute_drop_from_attentions(attentions, obs_window, top_k):
    """attentions: tuple of L layers, each [B, H, W, T]."""
    L = len(attentions)
    per_layer = []
    for a in attentions:
        if a.shape[2] != obs_window:
            window = a[0, :, -obs_window:, :]
        else:
            window = a[0]
        per_layer.append(head_agreement_layer(window, top_k))
    third = max(1, L // 3)
    early = sum(per_layer[:third]) / third
    late = sum(per_layer[L - third:]) / third
    return early - late, per_layer


def compute_drop_two_pass(base_model, ids, obs_window, top_k, device):
    """Two-pass: sdpa prefill (no attentions), then re-forward last obs_window
    queries in eager to extract attention pattern cheaply."""
    T = ids.shape[1]

    def switch_layers(impl):
        for layer in base_model.model.layers:
            try:
                layer.self_attn.config._attn_implementation = impl
            except AttributeError:
                pass

    loaded_impl = getattr(base_model.config, "_attn_implementation", "eager")

    # Pass 1: full prefill (sdpa) to get the K,V cache up to T-obs_window.
    with torch.no_grad():
        prefill = base_model(
            input_ids=ids,
            output_attentions=False,
            use_cache=True,
            return_dict=True,
        )
    past = prefill.past_key_values
    full_keys = [layer.keys.clone() for layer in past.layers]
    full_values = [layer.values.clone() for layer in past.layers]
    del prefill, past
    torch.cuda.empty_cache()

    # Pass 2: eager re-forward of last obs_window queries to capture attention.
    past_short = DynamicCache()
    for i, (k, v) in enumerate(zip(full_keys, full_values)):
        past_short.update(
            k[:, :, : T - obs_window, :].clone(),
            v[:, :, : T - obs_window, :].clone(),
            i,
        )
    last_ids = ids[:, -obs_window:]
    pos_ids = torch.arange(T - obs_window, T, device=device).unsqueeze(0)
    cache_pos = torch.arange(T - obs_window, T, device=device)

    switch_layers("eager")
    try:
        with torch.no_grad():
            scoring = base_model(
                input_ids=last_ids,
                past_key_values=past_short,
                position_ids=pos_ids,
                cache_position=cache_pos,
                output_attentions=True,
                use_cache=False,
                return_dict=True,
            )
    finally:
        switch_layers(loaded_impl)

    drop, per_layer = compute_drop_from_attentions(scoring.attentions, obs_window, top_k)
    del scoring, past_short, full_keys, full_values
    torch.cuda.empty_cache()
    return drop, per_layer


# ---------------- Correctness (mirrors gated_eviction.py) ----------------

def is_correct(pred, golds, task=None):
    pred_lower = pred.lower()
    if task in ("vt", "fwe", "cwe", "niah_multivalue"):
        return all(g.strip().lower() in pred_lower for g in golds)
    return any(g.strip().lower() in pred_lower for g in golds)


# ---------------- Prompt build ----------------

def build_prompt(tokenizer, ex):
    user_msg = ex["context"] + "\n\n" + ex["question"]
    if ex.get("answer_prefix"):
        user_msg = user_msg + "\n" + ex["answer_prefix"]
    messages = [
        {"role": "system", "content": "Answer the question concisely."},
        {"role": "user", "content": user_msg},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ---------------- DBTrimKV generation ----------------

def make_cache(num_layers, num_heads, memory_size, buffer_size=32, device="cuda"):
    return PagedTrimKVCache(
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=8192,
        min_tokens_per_head=0,
        strategy="fixed_budget",
        memory_size=memory_size,
        num_blocks_ratio=1.0,
        buffer_size=buffer_size,
        device=device,
    )


def generate_dbtrimkv(model, tokenizer, inputs, memory_size, do_compress, max_new):
    """Run DBTrimKV generation with compress enabled or disabled.

    do_compress=False -> the gate is closed and we let the paged cache grow
    unbounded (full KV equivalent through the DBTrimKV attention path).
    """
    cache = make_cache(
        num_layers=model.config.num_hidden_layers,
        num_heads=model.config.num_key_value_heads,
        memory_size=memory_size,
        device=model.device.type,
    )
    cache.do_compress = bool(do_compress)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new,
        do_sample=False,
        past_key_values=cache,
    )
    n_seen = cache._seen_tokens
    # n_kept = total tokens currently retained across all (layer, head)
    if cache.paged_cache is not None:
        n_kept = int(cache.paged_cache.cache_seqlens.sum().item())
    else:
        n_kept = n_seen * model.config.num_hidden_layers * model.config.num_key_value_heads
    return out, n_kept, n_seen


# ---------------- Main ----------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="4096", help="RULER context length")
    p.add_argument("--tasks", default="niah_multikey_3,vt,fwe,qa_1")
    p.add_argument("--per_task", type=int, default=30)
    p.add_argument("--budgets", default="128,256,512",
                   help="memory_size budgets for DBTrimKV (paper Fig 5)")
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--obs_window", type=int, default=32)
    p.add_argument("--top_k", type=int, default=32)
    p.add_argument("--max_new", type=int, default=32)
    p.add_argument("--out", required=True)
    p.add_argument("--db_model", default="ngocbh/DBTrimKV-Qwen3-4B-Instruct-2507")
    p.add_argument("--base_model", default="Qwen/Qwen3-4B-Instruct-2507")
    args = p.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")]
    tasks = args.tasks.split(",")
    device = "cuda"

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # ---- dataset ----
    print(f">> loading RULER {args.config}")
    ds = load_dataset("simonjegou/ruler", args.config, split="test")
    rows = []
    for task in tasks:
        sub = ds.filter(lambda r: r["task"] == task)
        # deterministic slice
        n = min(args.per_task, len(sub))
        for i in range(n):
            rows.append(sub[i])
    print(f">> built mixed suite: {len(rows)} rows across {tasks}")

    # ---- tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True,
                                              padding_side="left")

    # ---- base model (for gate signal only) ----
    print(f">> loading base model for gate signal: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval()

    # ---- DBTrimKV model ----
    print(f">> loading DBTrimKV model: {args.db_model}")
    db_model = TrimKVQwen3ForCausalLM.from_pretrained(
        args.db_model,
        torch_dtype=torch.bfloat16,
        load_trimkv_weights=True,
        download_from="huggingface",
        use_cache=True,
        device_map="cuda",
    )
    db_model.config._attn_implementation = "flash_attention_2"
    db_model.eval()
    print(f">> loaded DBTrimKV: L={db_model.config.num_hidden_layers}, "
          f"KV-heads={db_model.config.num_key_value_heads}")

    results_f = open(args.out, "w")
    t0 = time.time()
    n_written = 0
    for ex_i, ex in enumerate(rows):
        prompt = build_prompt(tokenizer, ex)
        inputs = tokenizer([prompt], return_tensors="pt", add_special_tokens=False).to(device)
        T = inputs.input_ids.shape[1]

        # ---- gate signal ----
        try:
            drop, per_layer = compute_drop_two_pass(
                base_model, inputs.input_ids, args.obs_window, args.top_k, device
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            drop, per_layer = 0.0, []
            print(f"  ! OOM on gate signal for ex {ex_i}, skipping (defaulting drop=0)")
        gate_open = bool(drop >= args.tau)

        max_new = int(ex.get("max_new_tokens") or args.max_new)
        max_new = max(max_new, args.max_new)

        for M in budgets:
            # plain DBTrimKV: always compress
            try:
                out_plain, n_kept_plain, _ = generate_dbtrimkv(
                    db_model, tokenizer, inputs, memory_size=M,
                    do_compress=True, max_new=max_new,
                )
                pred_plain = tokenizer.decode(
                    out_plain[0][T:].tolist(), skip_special_tokens=True
                ).strip()
            except Exception as e:
                pred_plain = f"[error: {type(e).__name__}: {str(e)[:80]}]"
                n_kept_plain = -1

            # gated DBTrimKV: compress iff gate_open
            try:
                out_gated, n_kept_gated, _ = generate_dbtrimkv(
                    db_model, tokenizer, inputs, memory_size=M,
                    do_compress=gate_open, max_new=max_new,
                )
                pred_gated = tokenizer.decode(
                    out_gated[0][T:].tolist(), skip_special_tokens=True
                ).strip()
            except Exception as e:
                pred_gated = f"[error: {type(e).__name__}: {str(e)[:80]}]"
                n_kept_gated = -1

            ok_plain = is_correct(pred_plain, ex["answer"], task=ex["task"])
            ok_gated = is_correct(pred_gated, ex["answer"], task=ex["task"])

            rec = {
                "id": ex_i,
                "task": ex["task"],
                "budget": M,
                "T": T,
                "drop": float(drop),
                "gate_open": gate_open,
                "tau": args.tau,
                "gold": ex["answer"],
                "pred_plain": pred_plain[:300],
                "pred_gated": pred_gated[:300],
                "correct_plain": bool(ok_plain),
                "correct_gated": bool(ok_gated),
                "n_kept_plain": int(n_kept_plain),
                "n_kept_gated": int(n_kept_gated),
            }
            results_f.write(json.dumps(rec) + "\n")
            results_f.flush()
            n_written += 1

        torch.cuda.empty_cache()
        gc.collect()

        elapsed = time.time() - t0
        rate = (ex_i + 1) / max(elapsed, 1e-9)
        eta = (len(rows) - ex_i - 1) / max(rate, 1e-9)
        if (ex_i + 1) % 1 == 0:
            print(f"  [{ex_i+1}/{len(rows)}] task={ex['task']} T={T} "
                  f"drop={drop:+.4f} gate_open={gate_open} "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s")

    results_f.close()
    print(f">> wrote {n_written} records to {args.out} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
