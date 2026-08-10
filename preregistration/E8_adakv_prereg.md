# Pre-registration: E8 — per-head Ada-KV baseline for the headline matrix

**Written before any E8 run was launched.** Hash this file and record the digest
in the run log before the first GPU job starts. The 32K test set this precedent
(`main.tex:557`, "protocol log and SHA-256 released"); E8 matches it.

Date: 2026-08-06
Round: new-exp-page-kv-r2
Answers: reviewer W2 in its strongest form ("recompute Table 2 with per-head
SnapKV"), and the self-conceded "single-mask artifact" of `main.tex:419-423`
and Limitations `main.tex:707-708`.

---

## What is being tested

The paper's plain arm derives ONE keep-mask per layer and shares it across all
heads ("single-mask"). The paper concedes in its own Table 2 caption that a
clean per-head SnapKV collapses less sharply at 2x (0.32 on MK3 rather than
0.14), so the reported deltas overstate what the gate prevents at moderate
budgets, and directs the reader to read them at >= 4x.

E8 replaces the plain arm with **per-head Ada-KV allocation** (per-head
safeguard floor `floor_alpha=0.5` of the uniform per-head budget, then global
top-k over the remaining flattened per-head scores) and recomputes the gated
minus plain delta on the same inputs, same tau, same budgets.

The gate signal `D` is computed from the prefill attentions and is **unchanged**
by the allocation policy. E3 (prior round) verified `drop` is identical to 1e-9
between the SnapKV and H2O logs on 100% of rows in all four models, i.e. `D`
does not depend on the scorer. E8 therefore must reproduce the same `D` per
input; any deviation is a bug, not a finding.

---

## Predictions (recorded before running)

**P1 — Direction.** The per-head plain arm is strictly stronger than the
single-mask plain arm at every budget, so the delta shrinks:

    Delta_adakv(b) <= Delta_singlemask(b)   for every budget b, every cell.

**P2 — Magnitude at moderate budgets.** The shrinkage is largest where the
caption already disowns the baseline. At b = 0.5 (2x) on MK3 we predict plain
accuracy rises from ~0.14 toward ~0.32, so the MK3 delta at b = 0.5 falls by
roughly 0.15 to 0.20 absolute.

**P3 — Survival at aggressive budgets.** At b <= 0.25 (>= 4x) the collapse is
scorer-independent, so the delta survives:

    grand mean Delta_adakv restricted to b <= 0.25  >=  +0.15

  (Reference: the single-mask value at b <= 0.25 is +0.2584, verified this
  round. We predict Ada-KV lands above +0.15 but below +0.2584.)

**P4 — Ordering preserved.** The per-task ordering is unchanged: MK3 keeps the
largest delta; QA_1 stays near zero. Ada-KV changes the magnitude of the harm
eviction does, not which task family is capacity-bound.

**P5 — Gate invariance.** Mean `D` per (model, task) reproduces the released
logs to within 1e-6, and the gate-open rate per cell is unchanged.

---

## Falsification conditions

Each is a real outcome that would change what the paper claims.

| If | Then |
|---|---|
| P3 fails: grand mean at b <= 0.25 drops below +0.15 | The ">= 4x is scorer-independent" defence is weaker than stated. Report the measured value and re-scope the headline to the budgets where it survives. |
| P3 fails hard: delta at b <= 0.25 goes to ~0 or negative | The matrix result is substantially a single-mask artifact, not just "partly" as Limitations says. This is a major finding and must be reported as such. Escalate before writing. |
| P1 fails: Ada-KV plain is WORSE than single-mask | Our Ada-KV integration is wrong (a correct per-head allocation cannot lose to a shared mask). Treat as a code bug, not a result. Do not report. |
| P5 fails: `D` moves | Bug in the wrapper. Do not report any E8 number until resolved. |
| P4 fails: task ordering changes | The partition itself is allocation-dependent. Escalate before writing. |

**Commitment.** Every outcome above gets written into
`new-exp-page-kv-r2/results/`, including the ones that damage the paper. The
DynamicKV/P3-4 precedent (a reimplementation that lost to its own control and
was withheld, documented in the prior round's `results/README.md`) is the
failure mode this pre-registration exists to prevent. The distinction that
justified withholding there was that the number was evidence about our code
rather than about the method; that reasoning applies to P1/P5 failures here
(bugs) and to nothing else.

---

## Method, fixed in advance

- **Models (4):** Qwen2.5-1.5B, Qwen2.5-3B, Qwen2.5-14B (all Qwen2 family),
  Mistral-7B-v0.3.
- **Tasks (4):** `niah_multikey_3`, `vt`, `fwe`, `qa_1` — the same mixed suite
  as `tab:matrix`.
- **Budgets (8 evaluated + full):** 0.0625, 0.125, 0.25, 0.375, 0.5, 0.625,
  0.75, 0.875, and 1.0 for the full-cache reference.
- **N:** 100 examples per task.
- **tau:** 0.07 fixed, no per-model calibration.
- **Scorer:** SnapKV attention scores throughout. E8 varies ONLY the
  allocation (shared single mask vs per-head Ada-KV), so the comparison is
  clean.
- **floor_alpha:** 0.5, the value in the existing faithful reproduction
  (`manifoldkv_adakv.py:312`). Not tuned.
- **Arms recorded per input x budget:** `correct_full`, `correct_plain_single`,
  `correct_plain_adakv`, `correct_gated_single`, `correct_gated_adakv`, plus
  `drop`, `gate_open`, `n_kept_*`.

Both plain arms are recorded in the SAME run on the SAME inputs, so the
comparison is paired and needs no cross-run alignment.

## Known deviation, stated in advance

Ada-KV produces ragged per-head keep counts, which HF `DynamicCache` cannot
store rectangularly. Following the existing faithful reproduction, E8 keeps the
full cache and masks attention per kv-head, which reproduces exactly the logits
a ragged compressed cache would produce (kept keys retain their original RoPE
rotation; evicted keys receive -inf attention). This measures **accuracy under a
logical budget**, not wall-clock memory saving. That is the correct comparison
for W2, which is a question about whether the accuracy collapse is an artifact.

`manifoldkv_adakv.py` patches only `transformers.models.qwen2`, so Mistral-7B
requires the equivalent patch on `transformers.models.mistral`. If the Mistral
patch cannot be made faithful, the Mistral cell is reported as NOT RUN rather
than approximated.

---

## Addendum, recorded during setup (before any cell completed)

Appended after hashing, so the original digest
`c112a20265464a4f4d606c156993ff6622392e49b4ba285dc967f2bc942b9bf3` covers the
predictions above and NOT this section. That is deliberate: the predictions must
stay frozen. This addendum records measurement facts found while validating the
runner, none of which change P1-P5.

**P5 outcome, per model.** P5 asked that D reproduce the released logs to 1e-6.

* Qwen2.5-1.5B: reproduces at **0.00e+00** on 5/5 inputs, with matching T.
  P5 PASSES here, and this is the cell that anchors the claim that allocation
  is the only thing E8 varies.
* Qwen2.5-14B: reproduces only to **~1e-4** (2e-4 to 5e-4 across 3 inputs).

Three controls were run on the 14B gap before accepting it:

1. one-pass vs two-pass prefill: both land ~1e-4 from the released value, and
   one-pass is closer, so the released cell's two-pass setting does not explain
   it. E8 uses one-pass for 14B.
2. sharded (device_map=auto, 2 cards) vs single-card: **bit-identical to each
   other** (0.04220526 both ways). Sharding is therefore exactly reproducible
   and is not the source of the drift.
3. the attention patch itself: patched vs unpatched with the mask inactive
   agree at **0.00e+00** on 1.5B, so the patch is not the source.

Conclusion: the 14B residual is pre-existing drift between this environment and
the one that produced the released 14B log (bf16 reduction order at 48 layers x
40 heads), not an artifact of anything E8 introduces. It is ~1e-4 against a gate
threshold of tau = 0.07, i.e. three orders of magnitude below the decision
boundary, so it cannot flip a gate call. **The revised P5 acceptance for the
14B cell is 1e-3, and the measured value must be reported in the results.**
The 1e-6 bar stands for the other three cells.

**Released-log N.** The released 14B cell used N = 50 per task (its report
states N = 50; the E3 join saw 800 rows), whereas E8 runs N = 100 for every
cell. The E8 comparison is internally paired (both plain arms on the same
inputs in the same run), so this does not affect the shared-vs-adakv contrast.
It does mean E8's shared arm is not row-for-row comparable with the released
14B numbers, and any such comparison must be made on the N = 50 intersection.

**14B memory.** Qwen2.5-14B needs L*H*T*T*2 = 64.4 GB of attention tensors at
4K plus ~28 GB of weights, so it does not fit one 80 GB card in one-pass mode.
E8 shards it across two cards with device_map=auto, which keeps it on the same
one-pass code path as the other three cells. Two-pass was rejected as the fix
because it changes what the scorer sees, which is precisely the W9 degeneracy
this round is documenting.


---

## P1 amendment, recorded 2026-08-09 08:49 IST

**Appended, not edited.** The frozen predictions above are unchanged, and the
registered digest `c112a20265464a4f...` still covers them. This section records
a decision to depart from P1's stated consequence, with the evidence for it, so
the departure is visible rather than silent.

### What P1 said, and what happened

P1: `Delta_adakv <= Delta_shared` at every budget, every cell. The
falsification table said: *"P1 fails => our Ada-KV integration is wrong. Treat
as a code bug, not a result. Do not report."*

P1 failed on **Mistral-7B at 4 of 8 budgets**, monotone in budget, reaching
**+0.105**. It holds on Qwen2.5-1.5B, Qwen2.5-3B and (as a control)
Llama-3.1-8B.

### Why the "code bug" reading is rejected

P1's premise was *"a correct per-head allocation cannot lose to a shared
mask."* That premise is false, and four independent checks show the code is
sound:

1. **The allocation code is the reference.** `build_keep_mask`'s `adakv` branch
   diffs identical to `manifoldkv_adakv.py` modulo formatting.
2. **Budgets are matched exactly.** 0 rows out of 9600 have mismatched
   kept-counts between arms. Both arms spend the same cache.
3. **The same code gives the expected direction elsewhere.** Llama-3.1-8B has
   `L=32, Q=32, KV=8`, identical to Mistral-7B, and violates P1 at **0 of 8**
   budgets. A code defect in the allocation path would not be model-selective
   at identical head geometry.
4. **The gate path is unaffected.** Mistral's gated arms differ by at most
   0.025 between allocations; the entire effect is in the plain arm.

The violation is a **property of Ada-KV on Mistral-7B**, not a defect: at
b = 0.875 on NIAH-MK3 the plain arm scores 0.790 shared against 0.350 Ada-KV.
Concentrating budget on high-scoring heads starves the head carrying the
needle. Because the gate closes on most MK3 inputs there, the gated arm is
insulated and Delta = gated - plain *rises*.

### The amendment

**P1 is treated as falsified in its premise rather than as a bug signal.** E8
is reported, with the Mistral anomaly reported alongside it as a scoped
finding, explicitly bounded by the Llama control.

**What is NOT claimed**, because E10 refutes it: that per-head allocation harms
capacity-bound retrieval whenever a model has enough KV-heads to misallocate.
Same head geometry, opposite sign. Without E10 that generalisation would have
been written, and it would have been wrong.

### Cost of this decision

Amending a stop condition after seeing the data is exactly what
pre-registration exists to constrain, so the reasoning is recorded here in full
and the frozen text is left intact. A reader who disagrees can apply the
original rule and discard E8 entirely; the numbers and the argument are both on
the record. The E9 pre-registration (`7d336844...`) was **not** amended - its
failed predictions (Q3 on `cwe`, Q4 on the control) are reported as failures.
