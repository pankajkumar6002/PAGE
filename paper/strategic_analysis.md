# Strategic analysis: paper as of 2026-06-03

Audience: us. Goal: maximize the probability of ICLR 2027 acceptance with the
smallest set of additional work. Not "what would make a perfect paper" but
"what is the marginal experiment / framing with the highest acceptance lift".

## 1. What ICLR reviewers will respect

**The gating result is genuinely strong.** Mistral-7B at 4K NIAH-MK3: plain
SnapKV collapses 99% to 0% as budget shrinks; gated holds **89% at every
budget**, Δ = +89pp at b=0.0625, with 10/100 false positives
(`gating_consolidated_report.md` headline table). This is the clearest single
result in the paper. It is a "the baseline breaks, our fix doesn't" demo on a
*known-difficult* task, on a *non-Qwen* architecture, with *zero retraining*
and a *transferred threshold* (τ=0.07 calibrated on Qwen 1.5B). Reviewers will
not have seen this kind of catastrophic-failure-prevention result on retrieval
under matched budget.

**The cleanest theoretical claim is the context-length scaling formula.**
$\rho_{\mathrm{KV}} \approx (1 - A_{\mathrm{full}}(M, T)) \cdot p_{\mathrm{recov}}(\rho_\mathcal{R}/\gamma)$ in `theory_h3.md` §4. The
formula simultaneously predicts (i) context-length amplification, (ii) model-
size saturation, and (iii) the re-entry of large models into the dilution
regime at long context. Empirical verification in `outline.md` §6 — Qwen3B 16K
FWE rebounds to ρ=0.32 exactly where headroom returns. This is a non-trivial
joint scaling prediction and it actually matches data. Bui has the dilution
mechanism; we have the *amplification scaling*. That distinction will land.

**Process credibility.** Pre-registered ρ threshold, audit script that checks
budget=1.0 against clean prefill, honest documentation of the Qwen3B-16K
calibration failure, honest scoping after the Bui literature reconcile. The
v0.8 problem statement and consolidated reports read like a *researcher's*
documentation, not a slide deck. Reviewers reward this in borderline cases.

## 2. The 3-5 most credible reviewer objections

### O1. "Your method is gated-SnapKV. The strong baselines are Bui (DBTrimKV), CapKV, IndexMem, Ada-KV, PyramidKV. Your headline +18pp could be SnapKV's weakness, not your strength."

**Defense with planned work.** The in-progress `--score_policy {h2o, streamingllm, pyramidkv}` runs (claude.md "Open work") and pulled-Bui TrimKV
reproduction directly answer this. If gating lifts at least 2 of these methods
by 3+ pp on the mixed suite, the orthogonality claim survives.

**Additional framing to neutralize.** Reframe the headline metric: report
**"Δ over the best baseline at matched budget"** rather than "Δ over plain
SnapKV". Even if gated-SnapKV doesn't beat ungated-CapKV in absolute terms,
*gated-CapKV beats ungated-CapKV* is the orthogonality result and that is the
publishable claim. This is the biggest reviewer-facing reframing we still owe.

### O2. "The partition reduces to: 'retrieval tasks fail, integration tasks help'. That's well-known folklore and doesn't need a new predictor."

**Defense with planned work.** Two existing pieces of evidence already answer
this. (1) The drop predictor distinguishes NIAH-MK3 from niah_multivalue and
niah_multiquery: both are "needle in haystack" syntactically, but only MK3 is
capacity-bound. Task-name-based heuristics fail this distinction; head-
agreement drop catches it (`theory_h3.md` §5 table). (2) The Qwen3B 16K FWE
result: same task family across contexts, but ρ moves from ~0 (4K saturated)
to 0.32 (16K) — folklore predicts neither.

**Additional experiment to neutralize.** Run the predictor on a task the
reviewer has never seen — *one synthetic NIAH where we continuously interpolate
between dilution-prone and capacity-bound* (see §3 below). If the drop tracks
the interpolation parameter smoothly, the folklore objection dies.

### O3. "The predictor doesn't transfer across architectures (SmolLM2 misranks; Qwen3B-16K needs τ=0). It works on Qwen and is shaky elsewhere."

**Defense with planned work.** The τ sensitivity sweep
(`gating_consolidated_report.md` §τ-sensitivity) shows the optimal τ lies in
[0.05, 0.20] across all 7 cells. The *ordering* of tasks by drop is preserved
on every cell, including the failure case. Frame the result as "the bare drop
is a universal *ordering statistic*; τ needs light per-(model, context)
calibration on a 50-input pilot."

**Additional experiment to neutralize.** A **one-shot calibration recipe**:
given any new (model, context), measure the drop on 20 inputs from a known
capacity-bound task (NIAH-MK3) and 20 from a known dilution-prone task (VT);
set τ to the midpoint. Show this beats the fixed τ=0.07 on the Qwen3B-16K
failure cell *and on all other cells* without hurting them. This is a 1-day
experiment and turns the predictor from "works most of the time" into "comes
with a recipe."

### O4. "Bui's Proposition 3.1 already establishes the dilution mechanism. CapKV gives a closed-form scorer. What's the theoretical contribution beyond those?"

**Defense with planned work.** Proposition 3 in `theory_h3.md` (context-length
amplification) goes beyond Bui's single-step bound by predicting joint
(M, T)-scaling of the recovery rate, and the empirical fit in `outline.md` §6
is concrete enough to defend.

**Additional theory needed (see §4 below).** This is the weakest piece of the
paper today. The honest answer is that the current theory section is mostly
*re-derivation with new notation*, plus a scaling sketch. We need one genuine
beyond-Bui claim to anchor the theory section; §4 picks the most realistic
target.

### O5. "Only 7 (model, context) cells, only RULER. The claim 'universal partition' is overfit."

**Defense with planned work.** LongBench cross-benchmark runs in flight
(claude.md "Open work"). If LongBench narrative_qa / hotpotqa / passkey
cluster correctly relative to the predictor on at least 3 models, the
universal claim is on much firmer ground.

**Additional experiment to neutralize.** A **single code or multilingual task**
added to the partition table — if drop predicts the right side for code
completion (intuitively dilution-prone: many distractor function bodies)
without any tuning, that closes the cross-domain gap with one extra figure.
Cheap relative to a multi-week benchmarking project.

## 3. The single most impactful experiment we haven't planned yet

**The synthetic dilution-to-capacity interpolation.** A single controllable
NIAH variant where one knob smoothly varies between "16-key NIAH-MK3 needs
exactly one needle" (capacity-bound) and "find the average / count of all
matching needles" (dilution-prone). Vary the knob in 7 steps. Measure (a) ρ at
each step under SnapKV sweep, (b) head-agreement drop at each step.

**Why this is the highest-leverage experiment.**

- It is a *predicted-curve* experiment, not a binary partition. If drop and ρ
  both move smoothly and monotonically with the knob, the predictor stops
  looking like "a binary task-family detector dressed up as a statistic" and
  starts looking like "a continuous measure of dilution headroom."
- It neutralizes O2 (folklore) and O4 (theory weakness) in one shot. The
  smooth interpolation is the experiment Bui can't run with his fixed-task
  suite, and it's exactly the empirical setup needed to justify the Δ→0
  asymptotic in our Proposition 3.
- It is cheap. One controllable synthetic dataset, one Qwen model, one A100.
  Estimated effort: 3-5 days including the dataset design.
- It produces a clean figure: one curve of ρ(knob), one curve of drop(knob),
  Pearson correlation in the title.

Reject the alternatives: per-head SnapKV would marginally tighten the upper
bound but reviewers don't reward "+2pp on tightening"; gated-DBTrimKV is
already in the open-work list and a comparison, not a new contribution; a
32K benchmark and production throughput are scope expansions that buy
reviewer goodwill but no new mechanism. The smoothness test buys mechanism.

## 4. The single biggest theoretical move beyond Bui

**Realistic target: prove a sufficient condition on the head-agreement drop
that guarantees $\rho > 0$ under good scoring.** The structure of the claim:

> Let $\delta_t$ be Bui's dilution quantity at step $t$. Let $D$ be the early-to-
> late head-agreement drop. Under noise-isotropy and a good-scoring assumption
> with retention ratio $\gamma$, there exist constants $c_1, c_2$ depending
> only on head count $H$ and depth $L$ such that
> $$D \geq c_1 \log(1/\gamma) \;\Longrightarrow\; \mathbb{E}[\delta_t] \geq c_2 (1 - A_{\mathrm{full}}).$$

In plain English: a large head-agreement drop is a sufficient statistic for
the dilution Bui's mechanism predicts. The drop becomes a *measurable proxy*
for the quantity $\alpha_\mathcal{R}$ in Bui's framework, which is the
unobservable Bui needs to assume away.

**Why this is the right beyond-Bui move.**

- It is the missing bridge between Bui's mechanism (assumes
  $\alpha_\mathcal{R}$ is known) and our predictor (measures something at
  inference). Today these are two unconnected results; the proof connects them.
- It implies a generalization-style claim: any architecture in which heads
  satisfy the agreement-drop bound inherits the partition.
- It is *probably provable* under the same surrogate model `theory_h3.md` §1
  already uses, with an extra step linking head divergence to attended-set
  size. The hard part is honest accounting of constants; the easy part is
  the structure.

**Compared to the other theory moves the user suggested:**

- A regret bound assumes a comparator class. The comparator is "the oracle
  partition," which is exactly what we're trying to characterize. Circular.
- A sample complexity for τ calibration is a *practical* result we could write,
  but a calibration recipe (see O3 above) is the same thing without a proof.
- A generalization bound on the partition (PAC-style) presumes the partition
  is sharp; we already know it is approximate (Qwen3B-16K). Would feel forced.

The sufficient-condition route is the only one that genuinely closes a gap
Bui left open and that the empirical evidence in hand already supports. Two
weeks of focused effort, including writeup. If the proof breaks honestly, the
fallback is a *proof sketch + numerical verification on the simplified
attention model* in `theory_h3.md`, which is still a step beyond what's there.

## 5. Smaller improvements ranked by impact / cost

- **Reframe headline from "Δ over plain SnapKV" to "Δ over best base evictor at
  matched budget"** — costs 1 day of replotting; doubles the
  defensibility of orthogonality claim. **Highest ratio.**
- **Calibration recipe** (one-shot, 20+20 inputs, midpoint τ) — 1 day; turns
  Qwen3B-16K from blemish into a documented protocol.
- **One LongBench task with the partition predictor table** — 1 day given the
  in-flight runs; directly attacks O5.
- **Drop-vs-ρ scatter plot across all (task, model, context) cells with
  Pearson r** — half a day; the single most convincing visual for the
  predictor.
- **Latency / memory table for the gating step** (already estimated 50ms in
  the report) — half a day for a real measurement; pre-empts "is it free?"
  question. Low impact but trivial cost.
- **Audit Bui's actual reported numbers on RULER and put them in a comparison
  table** — half a day; the absence of this table is a red flag.
- **Per-head SnapKV** — 1 week to implement padded-cache version; +2pp
  baseline at best. **Skip unless reviewer demands it.**
- **Cross-lingual / code task** — 3-5 days for one task; defends O5 mildly.
- **Joules / token on A100** — 2 days; reviewer-neutral, table-filler. Skip
  unless space.
- **Jetson Orin energy table** — depends on hardware availability; high
  variance return. Skip absent confirmed access.

## Honest overall judgement

The paper is currently at solid-borderline. The gating result is a real
ICLR-grade headline; the theory is the weak link. Without the
beyond-Bui theoretical move (§4) and without the orthogonality reframing
(O1, top of §5), this is a reject-or-weak-accept in a competitive year.
With those two, it is a defensible weak-accept. The one experiment in §3
plus the calibration recipe in §5 are what push it toward accept. Doing
more eviction baselines, more models, more contexts, more benchmarks does
not move the dial — those are already a credible 8-cell × 4-baseline
matrix and reviewers will not reward an 8-cell × 6-baseline matrix more
than they already do. The next 6 weeks should be spent on theory (§4),
the interpolation experiment (§3), and the headline reframing (O1) —
not on broader sweeps.
