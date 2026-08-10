# Pre-registration: E10 — Llama-3.1-8B as a bias control for the E8 P1 violation

**Written before the run.** Hash recorded before the GPU job starts, matching
E8 (`c112a202…`) and E9 (`7d336844…`).

Date: 2026-08-08
Round: `new-exp-page-kv-r2`

## What is being discriminated

E8's pre-registered P1 (`Δ_adakv ≤ Δ_shared` at every budget, every cell)
failed at 7 of 24 points, **all on Mistral-7B**, with the excess growing
monotonically to +0.10. Both Qwen cells behaved as predicted.

The diagnosis so far: Ada-KV redistributes budget across **KV-heads**. Qwen2.5
-1.5B and -3B have only **2** KV-heads, so there is almost nothing to
redistribute. Mistral-7B has **8**, and there the plain arm loses **−0.42** on
NIAH-MK3 at b = 0.875 while being neutral elsewhere.

**Two explanations remain confounded**, because in the current 3-cell design
"8 KV-heads" and "Mistral" are the same cell:

* **H1 — architectural**: any model with 8 KV-heads misallocates under Ada-KV,
  starving the single decisive head on capacity-bound inputs.
* **H2 — model-specific**: something particular to Mistral-7B, not the
  KV-head count.

**Llama-3.1-8B separates them.** It has L=32, Q=32, **KV=8**, GQA ratio 4 —
identical in shape to Mistral-7B — but is a different architecture family with
different training. It is also already in the paper as a fourth family, so it
is not a new model introduced to rescue a result.

## Predictions (recorded before running)

**L1 — the discriminating call.** If H1 holds, Llama-3.1-8B shows the **same
sign** of P1 violation as Mistral: `Δ_adakv > Δ_shared` at moderate-to-high
budgets (b ≥ 0.5), with the excess growing in b. If H2 holds, Llama behaves
like the Qwen cells: `Δ_adakv ≤ Δ_shared` throughout.

**L2 — the mechanism, if H1.** Under H1 the plain-arm damage is concentrated on
`niah_multikey_3` and is negative at high budget:
`plain_adakv(MK3, b=0.875) < plain_shared(MK3, b=0.875)` by at least 0.10
absolute. Other tasks stay within ±0.05.

**L3 — gate insulation.** Whatever P1 does, the two **gated** arms stay close
(within 0.05 at every budget), because the gate closes on most MK3 inputs and a
closed gate holds the full cache in both arms.

**L4 — no free lunch on transfer.** Fixed τ = 0.07 is documented as failing on
the Llama family (`main.tex` §8). Mean D per task is reported but **no
capacity-bound claim is made from this cell**; E10 is about the allocation
comparison only.

## Falsification and what each outcome means

| outcome | reading |
|---|---|
| Llama violates P1 like Mistral (L1 + L2 hold) | **H1 confirmed.** The effect is architectural: per-head Ada-KV harms capacity-bound retrieval whenever there are enough KV-heads to misallocate. This is a reportable finding about Ada-KV, and P1's premise ("per-head allocation cannot lose") is simply false. Amend P1 openly. |
| Llama behaves like Qwen (no violation) | **H2 supported.** The Mistral result is model-specific and cannot be generalised. E8 should report Mistral as an anomaly needing its own explanation, and the aggregate becomes harder to defend. |
| Llama violates P1 but the damage is NOT on MK3 (L1 holds, L2 fails) | Mixed. The allocation effect is architectural but the single-decisive-head story is wrong. Escalate before writing. |
| L3 fails (gated arms diverge) | Something is wrong with the gate path, not the allocation. Treat as a bug and do not report E10. |

**Commitment.** All four outcomes get written into
`new-exp-page-kv-r2/results/`, including the H2 outcome that would weaken E8.
This control exists specifically to reduce the bias of drawing an architectural
conclusion from a single model.

## Method, fixed in advance

* **Model:** `meta-llama/Llama-3.1-8B-Instruct`. Nothing else changes.
* **Tasks:** `niah_multikey_3, vt, fwe, qa_1` — the same suite as E8.
* **Budgets:** 0.0625, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875.
* **N:** 100 per task. **τ:** 0.07. **floor_alpha:** 0.5, unchanged, not tuned.
* **Runner:** `adakv_matrix.py`, unchanged except that
  `transformers.models.llama` is now added to `patch_families()`. That addition
  is mandatory and was verified before the run: without it the per-head mask
  silently does not apply and both arms would be identical, which would
  masquerade as an H2 result.

## Known constraint, stated in advance

There is **no released-log reference** for a Llama Ada-KV run, so the
`RELEASED_PLAIN_MK3` guard cannot fire on this cell. The released
`gated_4k_llama31_8b.jsonl` does exist, so the run's `D` values and full-cache
accuracy can still be checked against it, and that check is the substitute for
the missing guard. If `D` does not reproduce, E10 is void.
