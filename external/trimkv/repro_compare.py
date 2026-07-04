"""
Compare DBTrimKV PagedTrimKVCache vs full KV cache on the same prompt,
to confirm the gate is actually being applied and to log inference cost.
"""

import os, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
import torch
torch.set_grad_enabled(False)

from transformers import AutoTokenizer, DynamicCache
from trimkv.models.qwen3 import TrimKVQwen3ForCausalLM
from trimkv.cache_utils import PagedTrimKVCache

MODEL_PATH = "ngocbh/DBTrimKV-Qwen3-4B-Math"

print(">> Loading DBTrimKV checkpoint")
model = TrimKVQwen3ForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    load_trimkv_weights=True,
    download_from="huggingface",
    use_cache=True,
    device_map="cuda",
)
model.config._attn_implementation = "flash_attention_2"
model.eval()
tok = AutoTokenizer.from_pretrained(model.config.base_model, use_fast=True, padding_side="left")

# Build a 4k-token NIAH-style prompt by repeating noise + a single needle
import random, uuid
random.seed(42)
N_PAIRS = 64
pairs = [(str(uuid.uuid4()), str(uuid.uuid4())) for _ in range(N_PAIRS)]
target_idx = 31
tk, tv = pairs[target_idx]
haystack = "\n".join(f"key {k} -> value {v}" for k, v in pairs)
prompt = (
    "Here are some key->value pairs. Each is a UUID.\n"
    f"{haystack}\n\n"
    f"Question: what is the value for key {tk}? Reply with only the value UUID."
)
text = tok.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
inp = tok([text], return_tensors="pt").to(model.device)
N = inp.input_ids.shape[1]
print(f">> Input tokens: {N}")
print(f">> Target value (first 12 chars): {tv[:12]}")


def run(label, past_key_values, max_new=40):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = model.generate(
        **inp,
        max_new_tokens=max_new,
        do_sample=False,
        past_key_values=past_key_values,
    )
    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    new_ids = out[0][N:].tolist()
    gen = tok.decode(new_ids, skip_special_tokens=True).strip()
    print(f"\n[{label}]  time={elapsed:.2f}s  new_tok={len(new_ids)}  "
          f"tok/s={len(new_ids)/max(elapsed,1e-9):.1f}  peak_mem={peak:.2f}GB")
    print(f"  output: {gen[:140]}")
    print(f"  contains target prefix ({tv[:12]}): {tv[:12] in gen}")
    print(f"  exact target match: {tv in gen}")
    return gen, elapsed, peak


# NOTE: a plain DynamicCache can't be used as baseline here — the TrimKV-modified
# attention forward expects past_key_values.update(...) to return the 5-tuple
# (k, v, retention_weights, kv_positions, flash_attn_kwargs). We compare two
# budgets instead.

# 2) DBTrimKV @ memory_size=128 (paper default for DBTrimKV-Math)
print("\n=== DBTrimKV PagedTrimKVCache (memory_size=128) ===")
db128 = PagedTrimKVCache(
    num_layers=model.config.num_hidden_layers,
    num_heads=model.config.num_key_value_heads,
    max_seq_len=8192,
    min_tokens_per_head=0,
    strategy="fixed_budget",
    memory_size=128,
    num_blocks_ratio=1.0,
    buffer_size=32,
    device="cuda",
)
run("DBTRIMKV_M128", db128)

# 3) DBTrimKV @ memory_size=512 (looser budget)
print("\n=== DBTrimKV PagedTrimKVCache (memory_size=512) ===")
db512 = PagedTrimKVCache(
    num_layers=model.config.num_hidden_layers,
    num_heads=model.config.num_key_value_heads,
    max_seq_len=8192,
    min_tokens_per_head=0,
    strategy="fixed_budget",
    memory_size=512,
    num_blocks_ratio=1.0,
    buffer_size=32,
    device="cuda",
)
run("DBTRIMKV_M512", db512)
