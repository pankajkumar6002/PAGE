# Llama-3.1-8B architecture bias control for the Ada-KV P1 violation

Written 2026-08-09 08:44 IST. Pre-registration `llama_architecture_bias_prereg.md`,
sha256 `1dd0094de0b1bc0b...`, verified at launch before any data existed.

## Why this control exists

The per-head Ada-KV matrix's pre-registered P1 (`Delta_adakv <= Delta_shared` at every budget, every
cell) failed **only on Mistral-7B**. The 3-cell design confounded two
explanations, because "8 KV-heads" and "Mistral" were the same cell:

* **H1, architectural**: any model with 8 KV-heads misallocates under Ada-KV.
* **H2, model-specific**: something particular to Mistral-7B.

Llama-3.1-8B is `L=32, Q=32, KV=8` — **identical in shape to Mistral-7B**, a
different family and training run, and already a fourth family in the paper.

## Result: H1 is refuted

| | Mistral-7B | Llama-3.1-8B |
|---|---|---|
| shape | L=32, Q=32, KV=8 | **identical** |
| P1 violations | **4 of 8 budgets** | **0 of 8** |
| plain-arm MK3 at b=0.875 (Ada-KV − shared) | **−0.440** | **+0.140** |

Same head geometry, opposite behaviour, and not marginally: Ada-KV *destroys*
MK3 accuracy on Mistral (0.790 → 0.350) and *improves* it on Llama
(0.850 → 0.990).

**L1 fails** (registered: Llama violates like Mistral if H1 holds). Llama
violates at 0 of 8 budgets; its excess is negative and grows *more* negative
with budget, which is exactly what P1 predicted.

**L2 fails decisively.** It predicted plain-arm damage on MK3 of at least
−0.10 under H1. Measured: **+0.140**, the opposite sign.

**Conclusion: the Ada-KV harm is Mistral-specific (H2), not a consequence of
KV-head count.** Any claim that per-head allocation harms capacity-bound
retrieval "whenever there are enough KV-heads" is unsupported.

## L3 fails, and the reason is already documented

L3 registered that the two *gated* arms stay within 0.05, on the logic that a
closed gate holds the full cache in both arms.

| cell | max abs gated difference | L3 |
|---|---:|---|
| Mistral-7B | 0.0250 | HOLDS |
| Llama-3.1-8B | **0.1200** | **FAILS** |

Diagnosed, not a gate-path bug. Llama gate-open rates at tau = 0.07:

| task | gate open |
|---|---:|
| `niah_multikey_3` | **54/100** |
| `vt` | 100/100 |
| `fwe` | 100/100 |
| `qa_1` | 96/100 |

The gate **opens on 54% of MK3 inputs**, so the gated arm is not insulated and
inherits the plain-arm difference. This is the fixed-tau transfer failure on
the Llama family that the paper already documents (main.tex S8, App. A.8), not
a defect in the gate path. L3's premise assumed the gate mostly closes on MK3,
which is true on Qwen and Mistral and false on Llama.

## Independent corroboration from Qwen2.5-14B (2026-08-11 01:18 IST)

The per-head Ada-KV matrix's Qwen2.5-14B cell completed after this control was written and was **not
designed as a test of it**. It violates P1 at **0 of 8** budgets.

Across all four per-head Ada-KV models the violation is confined to Mistral-7B alone —
**4 of 32 cell-budget points**:

| cell | KV-heads | P1 violations |
|---|---:|---:|
| Qwen2.5-1.5B | 2 | 0/8 |
| Qwen2.5-3B | 2 | 0/8 |
| Qwen2.5-14B | 8 | **0/8** |
| Mistral-7B | 8 | **4/8** |
| Llama-3.1-8B (this control) | 8 | **0/8** |

**Three models with 8 KV-heads, and only Mistral misbehaves.** That is a
second, unplanned refutation of the KV-head-count hypothesis.

## Caveats

* Llama `D` reproduces its released log to **~1.6e-3**, not exactly. Same class
  as the ~1e-4 residual on Qwen2.5-14B (bf16 reduction order), and ~40x below
  tau = 0.07, so it cannot flip a gate call except within 1.6e-3 of the
  threshold.
* There is **no released Ada-KV reference for Llama**, so the `shared/plain`
  guard cannot fire on this cell. Full-cache MK3 accuracy was checked against
  the released log and matches (1.000 vs 1.000).
* This cell makes **no capacity-bound claim**. Fixed tau is documented as
  failing on this family; this control is an allocation comparison only.
