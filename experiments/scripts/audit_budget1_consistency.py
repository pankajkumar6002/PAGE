"""Verify that at budget=1.0 my eviction pipeline gives identical predictions
to a clean prefill-then-decode (no re-attention shenanigans).

If they differ at budget=1.0, the full-KV baseline in all reported tables
is biased and the relative rho numbers might still be OK but the per-budget
accuracy curves are off.
"""
import json
import sys

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


sys.path.insert(0, "experiments/scripts")
from ruler_sweep import (
    build_prompt,
    derive_keep_mask,
    generate_from_past,
    prefill_and_score,
    is_correct,
)


def clean_prefill_decode(model, tokenizer, ids, max_new, device):
    """Standard prefill then decode using HF's own logic. No re-attention,
    no manual position_ids; uses the prefill cache directly."""
    T = ids.shape[1]
    with torch.no_grad():
        prefill = model(input_ids=ids, use_cache=True, return_dict=True)
    past = prefill.past_key_values
    first_logits = prefill.logits[:, -1, :]
    next_token = first_logits.argmax(dim=-1, keepdim=True)
    out_ids = [int(next_token.item())]
    if int(next_token.item()) == tokenizer.eos_token_id:
        return tokenizer.decode(out_ids, skip_special_tokens=True).strip()

    for step in range(max_new - 1):
        with torch.no_grad():
            out = model(
                input_ids=next_token,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
        past = out.past_key_values
        new_id = int(out.logits[:, -1, :].argmax(dim=-1).item())
        out_ids.append(new_id)
        if new_id == tokenizer.eos_token_id:
            break
        next_token = torch.tensor([[new_id]], dtype=torch.long, device=device)

    return tokenizer.decode(out_ids, skip_special_tokens=True).strip()


def eviction_pipeline(model, tokenizer, ids, max_new, device, obs_window=32, n_sink=4):
    """Same code path as ruler_sweep at budget=1.0."""
    T = ids.shape[1]
    full_keys, full_values, _score = prefill_and_score(
        model, ids, obs_window, device, two_pass=False
    )

    keep_mask = torch.ones(T, dtype=torch.bool, device=device)
    keep_idx = keep_mask.nonzero(as_tuple=True)[0]
    past_b = DynamicCache()
    for i, (k, v) in enumerate(zip(full_keys, full_values)):
        past_b.update(k.index_select(2, keep_idx), v.index_select(2, keep_idx), i)
    return generate_from_past(model, tokenizer, past_b, ids, max_new, device, original_T=T)


def main():
    device = "cuda:0"
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device).eval()

    ds = load_dataset("simonjegou/ruler", "4096", split="test")

    print("checking budget=1.0 equivalence on 6 examples across tasks")
    print("task | T | clean correct | eviction correct | preds match?")
    print("---|---|---|---|---")
    n_match = 0
    n_check = 0
    for task in ["niah_multikey_3", "vt", "fwe", "qa_1", "qa_2", "niah_multivalue"]:
        sub = ds.filter(lambda r: r["task"] == task).select(range(1))
        ex = sub[0]
        prompt = build_prompt(tokenizer, ex)
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        T = ids.shape[1]
        max_new = int(ex.get("max_new_tokens") or 128)
        max_new = max(max_new, 128)

        clean_pred = clean_prefill_decode(model, tokenizer, ids, max_new, device)
        evict_pred = eviction_pipeline(model, tokenizer, ids, max_new, device)

        clean_ok = is_correct(clean_pred, ex["answer"], task=task)
        evict_ok = is_correct(evict_pred, ex["answer"], task=task)
        match = clean_pred == evict_pred
        n_check += 1
        if match:
            n_match += 1

        print(f"{task} | {T} | {clean_ok} | {evict_ok} | {match}")
        if not match:
            print(f"  clean : {clean_pred[:120]!r}")
            print(f"  evict : {evict_pred[:120]!r}")

    print(f"\nmatched {n_match}/{n_check}")


if __name__ == "__main__":
    main()
