# Quarantined: decode-loop bug, 2026-08-06

These files were produced before the `generate()` fix and must not be analysed.

**Bug.** The decode loop re-forwarded the entire prompt on step 0 against an
already-populated KV cache, double-counting the prompt and desynchronising
`position_ids` / `cache_position`. Generations truncated early or came back
empty.

**How it was caught.** Full-cache accuracy on Qwen2.5-1.5B NIAH-MK3 was 0.35
against the released log's 0.65, and the plain arm's collapse curve flattened
to ~0.20 where the released log falls 0.47 -> 0.00. Several `pred_full`
values were empty strings and one was a truncated UUID.

**Why the gate looked fine anyway.** `D` is computed from prefill attentions,
which the bug did not touch, so `D` still reproduced at 0.00e+00. Gate
fidelity alone was not sufficient evidence that the run was sound.

**Fix.** `generate()` now mirrors `gated_eviction.generate_from_past` exactly:
seed with the last prompt token against the populated cache, carry explicit
`position_ids` and `cache_position`, append EOS before breaking.

**Guard added.** The runner now asserts full-cache accuracy against the
released per-task values before writing any row, so this class of bug fails
loudly instead of producing plausible-looking numbers.
