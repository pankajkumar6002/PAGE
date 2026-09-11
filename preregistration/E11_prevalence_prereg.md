# Pre-registration: E11 — prevalence of the capacity-bound class in realistic workloads

**Written before any E11 run was launched.** Hash this file and record the
digest in the run log before the first GPU job starts. Precedent: the 32K test
(`main.tex:557`, "protocol log and SHA-256 released"), E8, and E9.

Date: 2026-08-26
Round: new-exp-page-kv-r2
Answers: reviewer **W1 / Q1** (ICLR review) and **DA-2** (ARS panel), the single
hardest and most-convergent objection across both reviews:

> "The central scientific claim — that a distinct capacity-bound regime exists
> *in realistic workloads* at meaningful prevalence — rests on one synthetic
> task family plus one code-completion subtask. A coarse D-and-sensitivity
> survey over 10–15 subtasks ... would substantially strengthen (or
> appropriately bound) the significance claim."

E9 already answered the narrower "is it more than one RULER task" question
(found `niah_multiquery`, falsified `cwe`). E11 answers the broader question E9
did not: **over realistic workloads, what fraction `f` of inputs are
capacity-bound, and is the class more than {NIAH-MK3, niah_multiquery, lcc}?**

---

## Operational definition (frozen — copied verbatim from the released analyzer)

A subtask's per-input and task-level classification uses the *exact* rule already
implemented in `page-kv/.../analyze_realistic_workload.py`, so E11 introduces no
new discretion. For a subtask:

- `A_full` = full-KV accuracy (b = 1.0, `correct_plain`).
- `sensitivity` = fraction of full-KV-correct inputs **destroyed** by plain
  eviction at the most aggressive budget `b_min` (= `1 − retention@b_min`).
- `rho` = Pr_x[ full-KV wrong ∧ some b<1.0 correct ] (the dilution/recovery
  signal).
- `testable` ⟺ `A_full ≥ 0.15` (below this the accuracy floor makes the call
  meaningless → class `untestable(floor)`).
- **capacity-bound** ⟺ `testable ∧ sensitivity ≥ 0.40 ∧ rho < 0.05`.
- **mixed** ⟺ `testable ∧ sensitivity ≥ 0.40 ∧ rho ≥ 0.05`.
- **dilution-prone** ⟺ `testable ∧ rho ≥ 0.05 ∧ sensitivity < 0.40`.
- **evict-robust** ⟺ `testable ∧ rho < 0.05 ∧ sensitivity < 0.40`.

The **workload share `f`** reported by E11 is, per subtask, the fraction of
inputs that are *individually* capacity-bound: full-KV-correct AND destroyed by
plain eviction at `b_min` AND `D < τ` (gate closes). Pooled `f` is the
input-weighted mean of this over the in-range panel. `f` carries a Wilson 95%
interval. This is a coarse estimate by construction and is reported as such.

`τ = 0.07`, budgets and gate/observation parameters are the released defaults
(obs_window 32, n_sink 4, top_k 32), matching every other cell in the paper.

---

## Subtask panel (frozen)

Grouped by data provenance. "Ingest" = a released log already exists and is
re-analyzed with no new GPU run; "run" = a new GPU cell.

**Ingest (LongBench, released plain-schema logs — verified present with the
`correct_plain`/`correct_gated`/`drop` schema `analyze_realistic_workload.py`
needs):** qasper, multifieldqa_en, trec, triviaqa (from `longbench_qwen*.jsonl`,
N=120–200 each); lcc, repobench-p, hotpotqa (from `longbench_realistic_*_qwen14b.jsonl`);
passage_count, passage_retrieval_en (from `longbench_pass*_*.jsonl`). Nine
subtasks, **zero GPU**. Expected: lcc capacity-bound; passage_count a benign
false-close (D<τ but eviction-safe, per `main.tex:378`); the rest
dilution-prone/evict-robust.

> The CapKV-wrapped logs `capkv_longbench_{qasper,triviaqa}.jsonl` are a
> *different* arm structure and are NOT used for ingest; the plain-SnapKV
> `longbench_qwen*.jsonl` logs are the correct source for these two subtasks.
> (Corrected during pre-run audit.)

**Run (LongBench gap subtasks — configs verified present in `Xnhyacinth/LongBench`,
34 configs total):** 2wikimqa, musique (multi-hop), gov_report or multi_news
(summarization). Predicted: multi-hop dilution-prone; summarization
evict-robust/dilution-prone. New runs use the released realistic-workload budget
grid **{1.0, 0.5, 0.25, 0.125, 0.0625}** (5 budgets, matching the existing
`longbench_realistic_*` logs) for comparability, not the 8/9-point RULER grid.

**Run (agentic / tool traces — AgentLongBench, `ign1s/AgentLongBench`, MIT):**
the shortest-context split available. Task categories: Tool Response,
Environment Response, Final Guess.

> **Boundary constraint (frozen commitment).** AgentLongBench contexts start at
> **32K**, at/beyond PAGE's declared 4K/16K fitting range; the pre-registered
> 32K test already found constant-ratio proportionality fails there. Therefore
> AgentLongBench results are reported **out-of-fitting-range** and are **NOT
> pooled** into the in-range `f`. They contribute a gate-closure / D-ordering
> observation only (does the gate fire on agentic inputs; is D's task ordering
> intact), never an in-range prevalence number. If no AgentLongBench split at or
> below ~32K loads cleanly, the agentic slice is reported as "attempted, out of
> loadable range" rather than dropped silently or fabricated.

---

## Predictions (frozen before any run)

- **P1.** Pooled in-range `f` is small: point estimate `f ≤ 0.15`. (A *bounding*
  result strengthens the significance discussion either way; a large `f` would
  be a strong positive surprise.)
- **P2.** At least the code family (lcc and/or a second code subtask) classifies
  capacity-bound, reproducing the one known realistic member.
- **P3.** Multi-hop QA (2wikimqa, musique) classifies dilution-prone or mixed,
  NOT capacity-bound — distractor *count* does not drive the class (consistent
  with E9's `qa_1`/`qa_2` finding that near-tie surface form, not count, matters).
- **P4.** The gate's a-priori decision (D vs τ) matches the empirical class on
  ≥ 70% of in-range subtasks (one-sided predictor: reliable at *closing* on
  capacity-bound, weaker elsewhere — so misses are expected to be benign
  false-closes that cost compression, not accuracy).
- **P5.** On AgentLongBench (out of range), report only whether the gate closes
  (D < τ) on a majority of agentic inputs; no accuracy-class prediction is made.

**Falsification / escalation.** If P3 fails (a real-distractor multi-hop task is
capacity-bound), the paper's near-tie-surface-form mechanism account is
incomplete and must be revised — registered here as a substantive finding, not
a bug. If P1's `f` is large, the abstract's "driven primarily by one RULER task
family" clause must be softened (coupling noted in the revision plan).

---

## Guards

- Ingested cells reproduce their released `A_full` where a released reference
  exists (reuse the full-cache-accuracy check pattern from `adakv_matrix.py`);
  a mismatch aborts.
- Newly-run cells' logs must carry the runner's guard lines before their numbers
  enter `results/E11_prevalence.md`.
- The aggregator recomputes every headline number from raw jsonl independently
  of the per-subtask analyzer, and exits `CHECK: PASS` only if they agree.

## Addendum (measurement notes, appendable without invalidating the hash)

Everything above the "## Addendum" heading is frozen; the digest in
`E11_prevalence_prereg.sha256` covers only that frozen portion (same convention
as `paths.verify_prereg`, which strips from "## Addendum" onward before hashing).
