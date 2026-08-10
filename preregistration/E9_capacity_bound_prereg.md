# Pre-registration: E9 — is the capacity-bound class broader than one synthetic task?

**Written before any E9 run was launched.** Hash recorded in the run log before
the first GPU job starts, matching the precedent of the 32K test
(`main.tex:557`) and of E8 this round.

Date: 2026-08-06
Round: new-exp-page-kv-r2
Answers: reviewer **W1**, the single hardest objection in the review:

> "Every headline rests on NIAH-MK3. [...] So the predictor cleanly separates
> two-distractor inputs from one-distractor inputs, which is the contrast it
> was selected on. **The class the paper claims to detect is not established as
> a class.** Add at least two more capacity-bound tasks."

---

## The distinction this experiment turns on

The paper's mechanism is specifically **near-tie distractors**: distractors that
share the query's *surface form*. NIAH-MK3 embeds two UUIDs formatted exactly
like the target. `main.tex:1350-1361` reports the monotone gradient over
niah_multikey_1 (0 near-ties) -> _2 (1) -> _3 (2) and attributes `D` to
distractor count rather than task identity.

RULER also ships tasks with **real hard-negative distractors** that are *not*
near-tie: `qa_1` (20 real documents, 19 distractors, SQuAD-style) and `qa_2`
(30 documents, 29 distractors, multi-hop HotpotQA). These are topically
confusable but do not mimic the query's surface form.

Measured on the released Qwen2.5-1.5B log, `qa_1` has the **highest** mean `D`
of any task in the suite (+0.2362, gate opens on 100% of inputs) against
NIAH-MK3's +0.0437 (gate closes on 100%). So on the evidence already in hand,
**real distractor count does not drive `D` down; near-tie surface form does.**

E9 tests that claim rather than assuming it, and simultaneously looks for
capacity-bound tasks outside the NIAH-multikey family.

---

## Tasks, chosen by a-priori mechanism

| task | structure | prediction |
|---|---|---|
| `cwe` | common-words extraction; 10 exact words required, low redundancy | **capacity-bound** (D < tau) |
| `niah_multiquery` | 4 distinct magic numbers, all required | **capacity-bound** (D < tau) |
| `qa_2` | 30 real documents, 29 hard-negative distractors, multi-hop | **dilution-prone** (D > tau), despite having the most distractors of any task tested |
| `niah_single_1` | 1 needle, no distractors | **dilution-prone** (D > tau) — negative control |

Including two tasks predicted NOT to be capacity-bound is what makes this a
test rather than a search. `qa_2` is the discriminating case: it has by far the
most distractors, so a distractor-*count* account predicts a low `D`, while the
near-tie account predicts a high one.

---

## Predictions (recorded before running)

**Q1 — Ordering.** Mean `D` orders as:

    cwe, niah_multiquery  <  tau = 0.07  <  niah_single_1, qa_2

**Q2 — qa_2 is dilution-prone.** Mean `D`(qa_2) > 0.15 on Qwen2.5-1.5B, i.e.
comfortably above tau and in the same region as `qa_1` (+0.236). This is the
prediction that separates "near-tie surface form drives D" from "distractor
count drives D".

**Q3 — At least one new capacity-bound task.** At least one of `cwe`,
`niah_multiquery` has mean `D` < tau AND rho <= 0.05 (the paper's
pre-registered partition threshold), on at least 3 of 4 models.

**Q4 — Control fires.** `niah_single_1` has mean `D` > tau on every model. If a
task with no distractors at all reads as capacity-bound, the mechanism account
is wrong.

**Q5 — Gate follows D.** For any task with mean `D` < tau, the gated arm's
accuracy exceeds the plain arm's at b <= 0.25 (the gate protects what it
closes on).

---

## Falsification conditions

| If | Then |
|---|---|
| **Q2 fails**: qa_2 has low `D` and reads capacity-bound | The mechanism is distractor *count*, not near-tie surface form. The paper's Section on the distractor sweep needs rewriting. This would be a **substantive finding**, not a failure. Report it. |
| **Q3 fails**: neither cwe nor niah_multiquery is capacity-bound | The class really is narrow. Report as a boundary result: the predictor detects near-tie-distractor retrieval, and the paper should claim exactly that rather than a general "capacity-bound" class. Damaging but survivable. |
| **Q4 fails**: niah_single_1 reads capacity-bound | The distractor-gradient story in `tab:distractor` is wrong. **Escalate before writing anything.** |
| **Q1 partially holds** (one of two candidates qualifies) | Still a full answer to W1: one confirmed plus one falsified against a pre-registered prediction is stronger evidence of a real predictor than two confirmations. Report both. |
| A candidate qualifies on some models but not others | Report per-model. Do not aggregate away a transfer failure; the paper already has a documented one on the Llama family. |

**Commitment.** Every outcome above is written into
`new-exp-page-kv-r2/results/`, including outcomes that narrow the paper's
claim. The withheld DynamicKV comparison from the prior round is the failure
mode this exists to prevent; the only results that may be withheld are ones
traced to a bug in our own code, and that determination requires a named,
reproduced defect.

---

## Method, fixed in advance

- **Models (4):** Qwen2.5-1.5B, Qwen2.5-3B, Qwen2.5-14B, Mistral-7B-v0.3.
- **Tasks (4):** `cwe`, `niah_multiquery`, `qa_2`, `niah_single_1`.
- **Budgets:** 0.0625, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875 (+ full).
- **N:** 100 per task. **tau:** 0.07 fixed. **Scorer:** SnapKV.
- **Runner:** `adakv_matrix.py`, unchanged from E8, which records both the
  shared-mask and Ada-KV arms plus the full-cache reference on the same inputs.
- **Reported per task:** mean `D` with per-input spread, gate-closed fraction,
  rho with Wilson CI, gated-vs-plain delta, and tau-sensitivity over
  {0.04, 0.055, 0.07, 0.085, 0.10}.

## Known constraints, stated in advance

`cwe` (max_new 120), `niah_multiquery` (128), `qa_2` (32) and `niah_single_1`
(128) all pass through the canonical clamp `max_new = max(dataset, 128)`, which
a missing line already cost this round once on `vt`. The full-cache accuracy
guard in `adakv_matrix.py` has no released reference for these four tasks, so
for E9 it is **advisory only**: the guard cannot fire, and correctness rests on
the runner having been validated on the four released tasks first.

`qa_2` at 4K uses 30 documents in ~17k characters, which fits the 4096-token
config. No task here needs a context longer than the released cells used.