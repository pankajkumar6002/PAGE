# Quarantined: shared arm used a per-layer mask, 2026-08-06

Per-head Ada-KV runs produced before the `_global_snapkv_score` fix. Do not analyse.

**Bug.** The `shared` (plain baseline) arm pooled scores across heads but kept
them PER LAYER, so every layer derived its own keep-mask. The paper's actual
plain arm (`gated_eviction.pool_score`, policy `snapkv`) returns ONE `[T]`
score averaged equally across all layers, giving a single mask shared by every
layer and head.

The per-layer variant is strictly stronger: it lets each layer retain the
tokens that layer attends to. That made the baseline artificially good.

**Severity: ~10x on the headline task.** Qwen2.5-1.5B, NIAH-MK3, first 25
inputs:

| b | released plain | buggy shared (per-layer) | fixed shared |
|---|---|---|---|
| 0.25 | 0.040 | 0.440 | **0.040** |
| 0.5  | 0.080 | 0.475 | **0.080** |

The fixed arm reproduces the released log exactly.

**How it was caught.** Pre-registered prediction P1 (Ada-KV delta <= shared
delta) came back VIOLATED. Rather than override the stop condition, the
Ada-KV implementation was audited against `manifoldkv_adakv.py` (identical,
formatting only) and the shared arm against `gated_eviction.derive_keep_mask`
plus `pool_score` (different: per-layer vs global). Auditing found the defect
in the arm that had NOT been copied from a reference.

**Why the guard missed it.** The full-cache accuracy guard checks the
`correct_full` arm, which uses no mask at all, so a masking bug cannot move it.
`D` also reproduced exactly, since it comes from prefill attentions. Neither
signal constrains the plain arm. The released per-budget plain accuracies are
the only ground truth that does, and they are now checked.

**Consequence for the per-head Ada-KV result.** The earlier run reported
Ada-KV *losing* to the shared mask, which was an artifact of the inflated
baseline. With the fix, Ada-KV behaves as expected (0.360 vs 0.080 at b=0.5 on
MK3). Every number from these files is void.

---

**Late addition: `adakv_4k_qwen14b.jsonl` (569 rows).** The first `mv` of this
file raced the killed process's final flush, which recreated it in the live
results directory. Confirmed stale by two signals: mtime 21:44:49, before the
fixed run began at 21:45:34; and shared plain accuracy on NIAH-MK3 at b=0.5 of
**0.930 against a released 0.600**, the inflated per-layer signature. Moved
here. When killing a run, verify the output file is gone AFTER the process has
actually exited.
