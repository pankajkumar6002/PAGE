"""
Single-example reproduction of DBTrimKV inference on a NIAH-style
KV-retrieval task. Uses DBTrimKV-Qwen3-4B-Math (smallest LLM
DBTrimKV checkpoint shipped).

NIAH-MK3 is RULER's "needle-in-a-haystack, multi-key-value" task:
the model is given a haystack of UUID -> UUID pairs, then asked
the value for a specific key. We emulate this here with a small
synthetic prompt because we want a fast end-to-end smoke test.
"""

import os, sys, time, json
import torch

# We want a single GPU so we don't trip device_map
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  # GPU 1 has only 4.5 GB used
torch.set_grad_enabled(False)

from transformers import AutoTokenizer
from trimkv.models.qwen3 import TrimKVQwen3ForCausalLM
from trimkv.cache_utils import PagedTrimKVCache

MODEL_PATH = "ngocbh/DBTrimKV-Qwen3-4B-Math"

print(">> Loading DBTrimKV checkpoint:", MODEL_PATH)
t0 = time.time()
model = TrimKVQwen3ForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    load_trimkv_weights=True,
    download_from="huggingface",
    use_cache=True,
    device_map="cuda",
)
model.config._attn_implementation = "flash_attention_2"
print(f">> Model loaded in {time.time()-t0:.1f}s")
print(">> base_model:", getattr(model.config, "base_model", None))
print(">> retention_gate:", getattr(model.config, "retention_gate", None))
print(">> num_hidden_layers:", model.config.num_hidden_layers)
print(">> num_key_value_heads:", model.config.num_key_value_heads)

tokenizer = AutoTokenizer.from_pretrained(
    model.config.base_model, use_fast=True, padding_side="left"
)

# Build a small NIAH-MK3-style prompt: 32 UUID -> UUID pairs, ask for value of one key.
import random, uuid
random.seed(0)
N_PAIRS = 32
pairs = []
for _ in range(N_PAIRS):
    pairs.append((str(uuid.uuid4()), str(uuid.uuid4())))
target_idx = 17
target_key, target_value = pairs[target_idx]

lines = [f"key {k} -> value {v}" for (k, v) in pairs]
haystack = "\n".join(lines)
prompt = (
    "Here is a list of key->value mappings. Each key and value is a UUID.\n"
    f"{haystack}\n\n"
    f"Question: what is the value for key {target_key}?\n"
    "Answer with only the UUID, no extra text."
)
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
input_len = model_inputs.input_ids.shape[1]
print(f">> Input tokens: {input_len}")

# DBTrimKV cache: global budget, dynamically redistributed across heads
past_key_values = PagedTrimKVCache(
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

print(">> Generating with DBTrimKV cache...")
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=128,
    do_sample=False,
    past_key_values=past_key_values,
)
elapsed = time.time() - t0
peak_mem = torch.cuda.max_memory_allocated() / 1e9

output_ids = generated_ids[0][input_len:].tolist()
generated = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

print()
print("="*70)
print("RESULT")
print("="*70)
print(f"Target key:    {target_key}")
print(f"Target value:  {target_value}")
print(f"Model output:  {generated[:200]}")
print(f"Exact-match:   {target_value in generated}")
print(f"Wall time:     {elapsed:.2f}s   ({len(output_ids)} new tokens, "
      f"{len(output_ids)/max(elapsed,1e-9):.1f} tok/s)")
print(f"Peak GPU mem:  {peak_mem:.2f} GB")
# Cache stats
if hasattr(past_key_values, "peak_cached_tokens") and past_key_values.peak_cached_tokens is not None:
    print(f"Peak cached tokens (sum over layers/heads): {past_key_values.peak_cached_tokens}")
print(f"_seen_tokens:  {past_key_values._seen_tokens}")
