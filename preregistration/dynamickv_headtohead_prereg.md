# Pre-registration: E13 — DynamicKV adaptive-budget head-to-head

**Written before the head-to-head was analyzed.** Hash and record the digest in
the run log before the first GPU job. Precedent: 32K test, E8, E9, E11, E12.

Date: 2026-08-26
Round: new-exp-page-kv-r2
Answers: reviewer **EIC-4 / R2-1** (ARS panel) and **W4** (ICLR review):

> "The adaptive-budget family is the missing baseline, and it is not a minor
> omission. If an adaptive allocator can drive b → 1.0 on capacity-bound inputs,
> it already does what PAGE does. The paper must either benchmark one of these or
> argue explicitly why allocation and admission are non-comparable operations."

E13 benchmarks a per-layer adaptive-budget allocator (DynamicKV-style) against
the PAGE gate, using the existing runner `dynamickv_headtohead.py`.

---

## What is being tested (frozen)

The runner produces three arms per (input, budget), all at **matched memory**
(the adaptive arm redistributes a fixed total budget across layers; it does not
spend more):

- **plain** — DynamicKV-style per-layer adaptive budget, eviction always on.
- **uniform** — same scorer, uniform per-layer budget (isolates what the
  *adaptivity* itself buys, independent of admission).
- **gated** — PAGE gate wrapped around the plain arm (retain full cache when
  D < τ).

Row schema (already emitted by the runner): `task, id, budget, T, drop,
gate_open, correct_plain, correct_uniform, correct_gated, kept_plain,
kept_uniform`.

> **Implementation honesty (inherited from the runner's own docstring).** There
> is no public DynamicKV reference implementation; this is a reimplementation
> from the paper's description and is reported as "our DynamicKV-style
> reimplementation," with the `uniform` control alongside so the reader sees what
> the adaptivity buys rather than the reimplementation details.

**Panel**: RULER 4K, the four-task suite (niah_multikey_3, vt, fwe, qa_1),
budgets 1.0/0.5/0.25/0.125/0.0625, τ = 0.07. A **released run already exists**
for **Qwen3-4B** (`dynamickv_qwen3_4k.jsonl`, 1000 rows, 250/task, verified
b=1.0 self-consistency 0 violations), which the analyzer validates offline and
which is a legitimate cell in its own right (Qwen3 is one of the paper's
cross-family models). Additional models (Qwen2.5-3B, Mistral-7B) are optional
GPU runs via the same runner; they broaden the claim but the core P1/P2 test is
answerable on the existing Qwen3-4B log. (The runner default model is Qwen3-4B;
the earlier draft's Qwen2.5 model list was an unverified assumption, corrected in
the pre-run audit.)

## The claim it settles (frozen)

The reviewer's threat is "adaptive allocation already does what PAGE does." The
`uniform` vs `plain` contrast measures what allocation buys; the `plain` vs
`gated` contrast measures the residual *admission* value on top of allocation.

- **P1.** On the capacity-bound task (niah_multikey_3), adaptive allocation
  (plain) does **not** recover the collapse at aggressive budgets: plain MK3
  accuracy at b ≤ 0.125 stays far below full-KV, i.e. allocation alone does not
  drive the effective budget to full on capacity-bound inputs. Predicted plain
  MK3 acc at b = 0.0625 < 0.2 (vs full-KV ≈ its A_full).
- **P2.** The gate still adds value over adaptive allocation: gated − plain on
  MK3 at b ≤ 0.125 is strongly positive (predicted ≥ +0.3 on at least one
  model), because the gate closes and retains full cache where allocation cannot.
- **P3.** On dilution-prone tasks the gate does **no harm**: gated ≥ plain − 1/N
  at every budget (1/N = single-input granularity floor; a lone flip is noise). (Stated as a one-sided no-harm test, not "all arms within a
  few pp": on a model where the gate over-closes — e.g. Qwen3's fixed-τ failure,
  `main.tex:170` — gated sits *above* plain by retaining the full cache, which is
  protective. The harm we must rule out is gated < plain, so P3 tests only the
  negative side. Corrected in pre-run audit after the released Qwen3-4B log showed
  large *positive* gated−plain gaps that a spread test would mis-flag.)

**Falsification / escalation.** If P1 fails — adaptive allocation *does* recover
MK3 at aggressive budgets — then allocation and admission overlap materially and
the paper must reposition PAGE relative to adaptive allocators rather than treat
them as orthogonal. Registered in advance as a substantive finding.

## Guards

- Full-cache (b = 1.0) accuracy per task must reproduce the released full-KV
  numbers where a released reference exists; a mismatch aborts.
- At b = 1.0 all arms retain the full cache, so `correct_plain == correct_uniform`
  on every b = 1.0 row — asserted by the analyzer as a self-consistency check.
- The analyzer recomputes matched-memory curves from raw jsonl and exits
  `CHECK: PASS` only if they agree with the per-arm summaries.

## Addendum (measurement notes, appendable without invalidating the hash)

Frozen portion ends at "## Addendum"; the digest in
`dynamickv_headtohead_prereg.sha256` covers only the frozen portion.
