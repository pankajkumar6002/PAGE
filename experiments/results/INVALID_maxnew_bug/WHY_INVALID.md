# Quarantined: max_new_tokens clamp missing, 2026-08-06

`adakv_4k_qwen3b.jsonl` produced before the `max_new` fix. Do not analyse.

**Bug.** The runner used the dataset's `max_new_tokens` directly. The canonical
runner (`gated_eviction.py:549-550`) takes the LARGER of that value and
`--max_new` (128):

    max_new = int(ex.get("max_new_tokens") or args.max_new)
    max_new = max(max_new, args.max_new)          # <- this line was missing

RULER gives `vt` only 30 tokens, but its `answer_prefix` is ~80 characters.
A verbose model restates the prefix and hits the cap before emitting any of the
5 answer variables.

**How it was caught.** The full-cache accuracy guard added after the earlier
decode-loop bug: `vt` came back at 0.000 against a released 1.000, while
niah_multikey_3 (0.930), fwe (0.760) and qa_1 (0.840) all matched exactly.

**Scope.** Model-dependent, which is why it was not universal:
* Qwen2.5-3B  -- vt 0.000 vs 1.00. INVALID, re-run.
* Qwen2.5-1.5B -- vt 0.820 vs 0.82. Valid, terse answers fit in 30 tokens.
* Mistral-7B  -- vt 0.930 vs 1.00. Within tolerance, valid.

Only this file is quarantined. The qwen15b and mistral7b runs from the same
batch passed the guard on all four tasks and are retained.
