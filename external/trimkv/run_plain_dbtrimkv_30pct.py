"""Plain DBTrimKV at ~30%-kept budgets on RULER 4K (matched-memory follow-up).

Reuses the tested helpers in external/trimkv/run_gated_dbtrimkv.py but runs ONLY
the plain DBTrimKV policy (do_compress=True) -- no gate signal, no full-KV gated
path -- to get accuracy vs memory_size at budgets that bracket 30% kept-KV
(kept-fraction per head = memory_size / T, meanT ~= 3742 -> 30% ~= 1123).
"""
import gc, json, os, sys, time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

TRIMKV = "/home/smlab/projects/eff-nn/external/trimkv"
sys.path.insert(0, TRIMKV)

import torch
torch.set_grad_enabled(False)
from datasets import load_dataset
from transformers import AutoTokenizer

from run_gated_dbtrimkv import build_prompt, generate_dbtrimkv, is_correct
from trimkv.models.qwen3 import TrimKVQwen3ForCausalLM

OUT = "/home/smlab/projects/eff-nn/experiments/results/plain_dbtrimkv_qwen3_4k_30pct.jsonl"
BUDGETS = [1024, 1152, 1280]
TASKS = ["niah_multikey_3", "vt", "fwe", "qa_1"]
PER_TASK = 30
DB_MODEL = "ngocbh/DBTrimKV-Qwen3-4B-Instruct-2507"
BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
MAX_NEW = 32
device = "cuda"

print(">> loading RULER 4096")
ds = load_dataset("simonjegou/ruler", "4096", split="test")
rows = []
for task in TASKS:
    sub = ds.filter(lambda r: r["task"] == task)
    n = min(PER_TASK, len(sub))
    for i in range(n):
        rows.append(sub[i])
print(f">> built mixed suite: {len(rows)} rows across {TASKS}")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, padding_side="left")

print(f">> loading DBTrimKV model: {DB_MODEL}")
db_model = TrimKVQwen3ForCausalLM.from_pretrained(
    DB_MODEL, torch_dtype=torch.bfloat16, load_trimkv_weights=True,
    download_from="huggingface", use_cache=True, device_map="cuda",
)
db_model.config._attn_implementation = "flash_attention_2"
db_model.eval()
print(f">> loaded: L={db_model.config.num_hidden_layers}, KV-heads={db_model.config.num_key_value_heads}")

f = open(OUT, "w")
t0 = time.time()
n = 0
for ex_i, ex in enumerate(rows):
    prompt = build_prompt(tokenizer, ex)
    inputs = tokenizer([prompt], return_tensors="pt", add_special_tokens=False).to(device)
    T = inputs.input_ids.shape[1]
    max_new = max(int(ex.get("max_new_tokens") or MAX_NEW), MAX_NEW)
    for M in BUDGETS:
        try:
            out_plain, n_kept_plain, n_seen = generate_dbtrimkv(
                db_model, tokenizer, inputs, memory_size=M, do_compress=True, max_new=max_new)
            pred = tokenizer.decode(out_plain[0][T:].tolist(), skip_special_tokens=True).strip()
        except Exception as e:
            pred = f"[error: {type(e).__name__}: {str(e)[:80]}]"
            n_kept_plain, n_seen = -1, -1
        ok = is_correct(pred, ex["answer"], task=ex["task"])
        rec = {"id": ex_i, "task": ex["task"], "budget": M, "T": T,
               "n_seen": int(n_seen), "n_kept_plain": int(n_kept_plain),
               "gold": ex["answer"], "pred_plain": pred[:300], "correct_plain": bool(ok)}
        f.write(json.dumps(rec) + "\n")
        f.flush()
        n += 1
    torch.cuda.empty_cache(); gc.collect()
    el = time.time() - t0
    rate = (ex_i + 1) / max(el, 1e-9)
    eta = (len(rows) - ex_i - 1) / max(rate, 1e-9)
    print(f"  [{ex_i+1}/{len(rows)}] task={ex['task']} T={T} elapsed={el:.0f}s eta={eta:.0f}s", flush=True)
f.close()
print(f">> wrote {n} records to {OUT} in {time.time()-t0:.0f}s")
