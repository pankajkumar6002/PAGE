import argparse
import math
import os
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from load_vlmodel import load_model

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class RunResult:
    gen_length: str
    method: str
    budget: str
    mem_saving_pct: str
    batch: str
    throughput_tok_s: float
    tokens_gen: int
    decode_time_s: float


def parse_args():
    p = argparse.ArgumentParser()

    # ====== Original benchmark controls ======
    p.add_argument("--context-len", type=int, default=32768, help="Input prompt length in tokens")
    p.add_argument("--max-new-tokens", type=int, default=128, help="Tokens to generate per sequence")
    p.add_argument("--batches", type=int, nargs="+", default=[1], help="Batch sizes to test")
    p.add_argument("--steps", type=int, default=2, help="Timed steps per batch size")
    p.add_argument("--warmup", type=int, default=2, help="Warmup steps per batch size (not timed)")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    p.add_argument("--greedy", action="store_true", help="Use greedy decoding (default).")
    p.add_argument("--temp", type=float, default=None, help="Temperature (ignored when --greedy).")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--output_dir", default="results/", help="output dir")
    p.add_argument("--overwrite", action="store_true", help="Whether to overwrite existing results")

    # ====== New: model parameters ======
    p.add_argument("--method", type=str, default="dbtrimkv", help="KV method")
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-8B-Thinking", help="Model path or HF id")
    p.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                   choices=["auto", "eager", "flash_attention_2", "sdpa"],
                   help="Attention implementation")
    p.add_argument("--max_model_len", type=int, default=None, help="Max model length")
    p.add_argument("--download_from", type=str, default="huggingface",
                   choices=["local", "wandb", "huggingface"],
                   help="Where to download the model from")

    # ====== New: general compression parameters ======
    p.add_argument("--kv_budget", type=int, default=512, help="KV budget for compression")
    p.add_argument("--update_kv", action=argparse.BooleanOptionalAction, default=True,
                   help="Whether to update KV")
    p.add_argument("--buffer_size", type=int, default=128, help="Buffer size for compression")
    p.add_argument("--fixed_kv_budget", action=argparse.BooleanOptionalAction, default=True,
                   help="Set to False for a fair comparison with visual token prunning methods. If set to False, the actual KV budget will be determined dynamically based on the text length, which is num_text_tokens + kv_budget.")
    p.add_argument("--compress_strategy", type=str, default="alpha", help="Compression strategy to use")

    # ====== New: RKV-specific parameters ======
    p.add_argument("--window_size", type=int, default=32, help="Window size for compression")
    p.add_argument("--mix_lambda", type=float, default=0.1, help="Mix lambda for compression")
    p.add_argument("--retain_ratio", type=float, default=0.2, help="Retain ratio for compression")
    p.add_argument("--retain_direction", type=str, default="last", help="Retain direction for compression")
    p.add_argument("--divide_method", type=str, default="step_length", help="Method to divide input")
    p.add_argument("--divide_length", type=int, default=128, help="Length to divide input")
    p.add_argument("--compression_content", type=str, default="all", help="Content to compress")

    # ====== New: StreamingLLM parameter ======
    p.add_argument("--first_tokens", type=int, default=4, help="First tokens for compression")

    # Back-compat aliases
    p.add_argument("--model", type=str, help="Alias for --model_path")
    p.add_argument("--attn-impl", dest="attn_impl_alias", type=str, choices=["auto", "eager", "flash_attention_2", "sdpa"],
                   help="Alias for --attn_implementation")
    p.add_argument("--model_type", type=str, default="qwen3_vl", help="Model type for loading VL model")

    return p.parse_args()


def to_torch_dtype(d):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[d]


def resolve_model_args(args):
    model_id = args.model_path or args.model
    attn_impl = args.attn_implementation if args.attn_impl_alias is None else args.attn_impl_alias
    return model_id, attn_impl


def make_batch(processor, batch_size: int, context_len: int, device: str):
    filler_id = processor.tokenizer.eos_token_id or 0
    input_ids = torch.full((batch_size, context_len), fill_value=filler_id, dtype=torch.long)
    input_ids[:, -1] = processor.tokenizer.bos_token_id or (filler_id ^ 1)
    attn = torch.ones_like(input_ids)
    return {"input_ids": input_ids.to(device), "attention_mask": attn.to(device)}


@torch.no_grad()
def time_decode(model, processor, prepare_input_for_generation_fn, batch, max_new_tokens: int, steps: int, warmup: int, greedy: bool, temp: float, device: str):
    gen_kwargs: Dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        do_sample=not greedy,
        use_cache=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=None,
        temperature=temp if (not greedy and temp is not None) else None,
    )
    for _ in range(warmup):
        batch = batch.copy()
        batch = prepare_input_for_generation_fn(model, batch)
        _ = model.generate(**batch, **gen_kwargs)

    if device == "cuda":
        torch.cuda.synchronize()

    total_time_list = []
    total_tokens_list = []
    tps_list = []

    for _ in range(steps):
        torch.cuda.empty_cache()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_time.record()
        batch = batch.copy()
        batch = prepare_input_for_generation_fn(model, batch)
        _ = model.generate(**batch, **gen_kwargs)
        end_time.record()
        torch.cuda.synchronize()
        total_time = start_time.elapsed_time(end_time) / 1000.0  # seconds
        total_tokens = batch["input_ids"].size(0) * max_new_tokens 
        tps = total_tokens / total_time if total_time > 0 else float("nan")

        total_time_list.append(total_time)
        total_tokens_list.append(total_tokens)
        tps_list.append(tps)

    total_time = sum(total_time_list) / steps
    total_tokens = sum(total_tokens_list) / steps
    tps = sum(tps_list) / steps

    return total_time, total_tokens, tps


def describe_budget_and_saving(args):
    # Budget string like examples: "Fixed – 1024" or "Ratio – 34% – 2 785"
    budget_str = "–"
    mem_save = "–"
    if args.kv_budget is not None:
        budget_str = f"Fixed – {args.kv_budget}"
    elif args.retain_ratio is not None:
        pct = int(round(args.retain_ratio * 100))
        budget_str = f"Ratio – {pct}%"
    if args.retain_ratio is not None:
        mem_save = f"{round((1.0 - args.retain_ratio) * 100, 2):.2f}"
    return budget_str, mem_save


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    model_id, attn_impl = resolve_model_args(args)

    if args.method == 'seerattn':
        args.model_path = "SeerAttention/SeerAttention-Decode-Qwen3-4B-AttnGates"
    if 'trimkv' in args.method.lower():
        args.model_path = "ngocbh/DBTrimKV-Qwen3-VL-8B-Thinking"

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"benchmark_{args.method}_{args.kv_budget}kv_{args.context_len//1000}k_{args.max_new_tokens}gen.csv")
    if os.path.exists(out_path) and not args.overwrite:
        print("Output file already exists, please remove it first:", out_path)
        return

    device = args.device
    model, processor, prepare_input_for_generation_fn = load_model(args)

    results: List[RunResult] = []
    budget_str, mem_save = describe_budget_and_saving(args)

    for bsz in args.batches:
        batch = make_batch(processor, bsz, args.context_len, device)
        try:
            torch.cuda.empty_cache()
            total_time, total_tokens, tps = time_decode(
                model, processor, prepare_input_for_generation_fn, batch,
                max_new_tokens=args.max_new_tokens,
                steps=args.steps,
                warmup=args.warmup,
                greedy=args.greedy,
                temp=args.temp,
                device=device,
            )
        except Exception as e:
            print(f"Error with batch size {bsz}: {e}")
            total_time, total_tokens, tps = float("nan"), 0, float("nan")

        results.append(RunResult(
            gen_length=f"{args.context_len//1000*1}K",
            method=str(args.method).upper() if args.method else "FullKV",
            budget=budget_str,
            mem_saving_pct=mem_save,
            batch=f"{bsz}",
            throughput_tok_s=tps,
            tokens_gen=total_tokens,
            decode_time_s=total_time,
        ))
        print(f"[batch={bsz}] throughput={tps:,.2f} tok/s | tokens={total_tokens:,} | time={total_time:.2f}s")

    import pandas as pd
    df = pd.DataFrame([asdict(r) for r in results])
    df = df[[
        "gen_length", "method", "budget", "mem_saving_pct",
        "batch", "throughput_tok_s", "tokens_gen", "decode_time_s"
    ]]
    df.to_csv(out_path, index=False)
    print("\nSaved:", out_path)
    try:
        from tabulate import tabulate
        print(tabulate(df, headers="keys", tablefmt="github", floatfmt=".2f"))
    except Exception:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
