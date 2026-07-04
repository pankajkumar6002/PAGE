# Architecture-normalized head-agreement-drop predictor: does normalization fix cross-architecture transfer?

Reproduce all numbers: `.venv/bin/python experiments/scripts/normalized_predictor.py`

## Verdict (summary)

**Yes, with one precise scope restriction.** The per-model **pooled z-score of the drop
(variant b)** — `z = (D − μ_model) / σ_model`, with μ, σ from an *unlabeled* mixed pilot on the
target model — is the paper-ready architecture-invariant form of the predictor:

- One global threshold **θ_z = −0.69** (fit only on the 5 Qwen/Mistral calibration cells, never
  touching Llama-family data) separates NIAH-MK3 task means from qa_1/qa_2/vt/fwe task means in
  **all 7 model-cells** including the two held-out Llama-family cells. The raw drop manages only
  4/7 at any single threshold.
- On **Llama-3.1-8B** it fixes the transfer failure outright: tab:matrix gated Δ
  **+0.109 → +0.174**, at the cost of keeping slightly more KV (0.175 → 0.236 at b = 0.0625)
  because MK3 now closes more often (open 0.56 → 0.30) — which is exactly the intended repair.
- On **Yi-1.5-9B** it converts the gate from a de-facto full-KV fallback (74–79% of KV kept)
  into a real compressor (22–32% kept; vt/fwe/qa gate open ≈ 1.00) while still improving over
  plain SnapKV at every budget (tab:matrix Δ **+0.145** vs paper's +0.390). The accuracy give-back
  is a genuine accuracy-compression trade-off, not a predictor error (see §5.3).

**Scope restriction:** no within-model monotone transform can put `niah_multivalue` above MK3 on
Yi or Llama, because its *mean drop is below MK3's* in both models (Yi −0.013 vs +0.019; Llama
+0.064 vs +0.073). The invariance claim must therefore be stated as *MK3 vs {qa_1, qa_2, vt,
fwe}*, with multivalue called out as behaviorally eviction-fragile on Llama-family models —
where its low score is arguably the *correct* per-input prediction (gating it closed gains
+0.36–0.70 at b = 0.0625 on both models).

**Shape-based variants (e) fail.** They do not merely underperform — they *degrade* the
per-input ranking, so the early-vs-late difference in raw agreement units is the right base
statistic; only its scale/offset is architecture-dependent.

---

## 1. Data and methodology

- **Seven model-cells** (per-input per-layer head-agreement probes):
  `drops_qwen15b_4k_n100.jsonl`, `drops_qwen3b_4k_n100.jsonl`, `drops_qwen3b_16k_n100.jsonl`,
  `drops_mistral7b_4k_n100.jsonl`, `drops_mistral7b_16k_n100.jsonl` (N≈100/task),
  `drops_yi15_9b_4k.jsonl`, `drops_llama31_8b_4k.jsonl` (N=50/task).
- **Per-input D** recomputed with the paper convention (`aggregate_drops_n100.py`):
  D = mean agreement over layers [0, L/3) − mean over [2L/3, L).
- Data hygiene: `drops_mistral7b_16k_n100.jsonl` contains an exact duplicate of its 100 MK3
  records (deduped) and is missing qa_2; each Mistral file has one truncated/blank line
  (skipped). Yi/Llama gated jsonl ids are global 0–299 in drops-file order.
- **Convention consistency** between the drops probes and the `drop` field stored in the gated
  jsonls: Yi max |stored − recomputed| = 1.1e−16 (identical). **Llama differs by up to 7.4e−3**
  (mean 3.2e−3; 17/300 inputs flip a τ = 0.07 decision) — the gated run's inline probe was a
  separate stochastic-order forward pass. All re-evaluations below use the *recomputed* D
  consistently; the old-gate columns use the stored decisions, so "old" exactly reproduces the
  paper numbers (+0.390 Yi, +0.109 Llama ✓).
- **Protocol:** normalization statistics use only the target model's own pooled inputs (no task
  labels for variants b–d). The global threshold is *fit on the 5 Qwen/Mistral cells only* and
  applied unchanged to the held-out Yi and Llama cells.

Variants tested (per-input, per-model):

| id | statistic | deployable? |
|---|---|---|
| raw | D, fixed τ = 0.07 | paper baseline |
| (a) | D / max_task(mean D) | **NO** — needs task labels; diagnostic only |
| (b) | z-score: (D − μ_pooled)/σ_pooled | yes — unlabeled pilot |
| (c) | min-max: (D − min)/(max − min) | yes — unlabeled pilot (fragile: extremes) |
| (d) | pooled quantile rank of D | yes — unlabeled pilot |
| (e) | shape: D/mean(a) ("relD"), −corr(a, depth), −slope/mean, core-drop (early bin skips first L/6), 1 − a_late/peak | yes — no pilot needed |

Deployability caveat for (b)–(d): the pilot must be an unlabeled *mixed* sample resembling the
deployment task mix (the paper's existing 40–50-input calibration recipe already assumes a
pilot; this variant drops that recipe's *label* requirement).

## 2. Shape variants (e): the hypothesis test that failed

Per-cell per-input AUC, MK3 (should be low) vs the other five tasks (should be high):

| statistic | Qwen1.5B-4K | Qwen3B-4K | Qwen3B-16K | Mistral7B-4K | Mistral7B-16K | Yi1.5-9B-4K | Llama3.1-8B-4K |
|---|---:|---:|---:|---:|---:|---:|---:|
| **rawD** | **1.000** | **1.000** | **1.000** | **0.958** | **0.973** | **0.803** | **0.810** |
| relD = D/mean(a) | 0.999 | 1.000 | 1.000 | 0.793 | 0.899 | 0.799 | 0.651 |
| −corr(a, depth) | 1.000 | 1.000 | 1.000 | 0.393 | 0.781 | 0.791 | 0.224 |
| −slope/mean | 1.000 | 1.000 | 1.000 | 0.667 | 0.853 | 0.798 | 0.487 |
| coreD (skip first L/6) | 1.000 | 1.000 | 1.000 | 0.694 | 0.733 | 0.775 | 0.643 |
| coreD/mean | 1.000 | 1.000 | 1.000 | 0.513 | 0.547 | 0.747 | 0.498 |
| 1 − a_late/peak | 0.934 | 0.899 | 0.942 | 0.842 | 0.755 | 0.770 | 0.616 |

Every shape statistic is ≤ rawD in every cell, catastrophically so exactly where invariance is
needed (depth-correlation AUC **0.224** on Llama — *inverted*: MK3 inputs decline more smoothly
with depth than dilution tasks there; 0.393 on Mistral-4K). The "different consensus dynamics"
hypothesis is real (the profiles do differ across families) but it contaminates shape features
rather than enabling them. Conclusion: keep rawD as the base statistic; normalize its
distribution. Since (a)–(d) are within-model monotone maps of D, their per-cell AUCs equal the
rawD row above — normalization can only fix the *cross-model threshold*, not within-model
ranking. That is sufficient: the failure mode was precisely the threshold.

## 3. Cross-architecture separation of the normalized variants

### 3.1 Best single global threshold across ALL SEVEN cells

Task-level criterion: MK3 task-mean below θ and the other task-means above θ, per cell.
"Strict" requires all 5 other tasks above; "relaxed" excludes niah_multivalue.
Balanced acc = ½(frac MK3 inputs below θ + frac rest inputs above θ), pooled over all cells.

| variant | strict: cells ok /7 | strict bal-acc | relaxed: cells ok /7 | relaxed bal-acc | pooled cross-cell AUC (excl mv) |
|---|---:|---:|---:|---:|---:|
| raw (τ sweep) | 3/7 | 0.837 | 4/7 | 0.859 | 0.904 (0.931) |
| (a) task-max | 3/7 | 0.800 | 4/7 | 0.819 | 0.903 (0.939) |
| **(b) z-score** | **5/7** | **0.925** | **7/7** | **0.955** | **0.965 (0.990)** |
| (c) min-max | 5/7 | 0.886 | 5/7 | 0.927 | 0.959 (0.984) |
| (d) quantile | 5/7 | 0.899 | 7/7 | 0.928 | 0.956 (0.983) |

- **Strict separation is impossible for any monotone variant** (max 5/7): on both Yi and Llama
  the multivalue mean sits below the MK3 mean, so the ordering itself is broken at task level.
  The 2 strict failures for z/minmax/quantile are exactly Yi and Llama, exactly via multivalue.
- Under the relaxed (mv-excluded) criterion, **z-score and quantile achieve 7/7**; z-score has
  the best balanced accuracy and pooled AUC (0.990 excl mv).
- The labeled "diagnostic upper bound" (a) is *not* an upper bound at all (4/7): dividing by the
  max task mean fixes scale but not offset, and the identity of the max task itself shifts
  (fwe on Llama). z-score fixes offset + scale and dominates it. (a) is also not deployable.

### 3.2 Held-out transfer: θ fit on Qwen/Mistral only

Fitting θ on the 5 calibration cells (relaxed criterion) gives **θ_z\* = −0.6934** for the
z-variant (calibration: 5/5 task-level, balanced acc 0.977). Applied unchanged to the held-out
cells:

| held-out cell | task-level sep (MK3 vs qa1/qa2/vt/fwe) | per-input open-frac MK3 | open-frac qa1/qa2/vt/fwe/mv | per-cell AUC (excl mv) |
|---|---|---:|---:|---:|
| Yi-1.5-9B-4K | **OK** | 0.34 | 1.00 (mv 0.02) | 0.998 |
| Llama-3.1-8B-4K | **OK** | 0.30 | 0.98 (mv 0.04) | 0.969 |

For comparison, quantile at its calibration-fit θ_q = 0.224 opens 0.64/0.54 of MK3 on Yi/Llama
(worse protection); min-max 0.82/0.62. **z-score is the best deployable variant** and is used
below.

Why fixed τ = 0.07 could never transfer, in one line — the z-position of 0.07 in each model's
pooled drop distribution:

| | Qwen1.5B-4K | Qwen3B-4K | Qwen3B-16K | Mistral7B-4K | Mistral7B-16K | Yi1.5-9B-4K | Llama3.1-8B-4K |
|---|---:|---:|---:|---:|---:|---:|---:|
| pooled μ_D | +0.134 | +0.059 | +0.059 | +0.082 | +0.089 | +0.046 | +0.089 |
| pooled σ_D | 0.062 | 0.039 | 0.043 | 0.016 | 0.028 | 0.036 | 0.019 |
| z(τ = 0.07) | −1.04 | +0.27 | +0.26 | −0.75 | −0.70 | **+0.69** | **−0.97** |
| raw equiv. of θ_z = −0.69 | 0.091 | 0.032 | 0.029 | 0.071 | 0.070 | 0.021 | 0.075 |

The same raw threshold sits at the 24th percentile-ish of Llama's distribution but *above the
mean* of Yi's — two opposite failure modes (over-open vs over-closed) fixed by one z-threshold.
Note the z-recipe reproduces ≈0.07 on the Mistral cells and ≈0.03 on Qwen-3B — it agrees with
the labeled midpoint calibration recipe where that recipe worked, and extends it label-free.

## 4. Exact gated-accuracy re-evaluation (variant b, θ_z = −0.69)

Setup: gate opens (evict at budget b) iff z ≥ −0.6934; closed keeps full KV. Full-KV outcomes:
Llama from budget = 1.0 rows; Yi (no b = 1.0 rows) from `correct_gated` of originally
gate-closed rows. Every input the new gate closes has a known full-KV outcome in both models
(0 unknowns), so the re-evaluation is **exact**, not approximate. "Oracle" = per-input gate
that opens exactly when plain eviction is correct (upper bound). Kept-KV = mean fraction of
the cache retained.

### 4.1 Llama-3.1-8B-Instruct, RULER 4K — the transfer failure is fixed

Gate-open fractions: MK3 0.56 → **0.30**, multivalue 0.12 → 0.04, qa/vt/fwe unchanged (0.96–1.00).

4-task tab:matrix suite (MK3 + vt + fwe + qa_1, N = 200):

| budget | plain | fixed-τ gated | **z-gated** | oracle | kept-KV fixed-τ → z |
|---:|---:|---:|---:|---:|---|
| 0.5 | 0.685 | 0.790 | **0.855** | 0.935 | 0.560 → 0.593 |
| 0.25 | 0.595 | 0.705 | **0.770** | 0.935 | 0.340 → 0.389 |
| 0.125 | 0.530 | 0.640 | **0.705** | 0.940 | 0.230 → 0.287 |
| 0.0625 | 0.225 | 0.335 | **0.400** | 0.935 | 0.175 → 0.236 |
| **grand-mean Δ vs plain** | | **+0.109** | **+0.174** | +0.427 | |

6-task pooled Δ: +0.089 → **+0.134**. Per-task grand means: MK3 0.460 → **0.720** (full-KV
0.860 as ceiling given 30% of MK3 still opens), multivalue 0.925 → 0.935, all four dilution
tasks unchanged. The fixed-τ failure ("MK3 mean D = 0.073 > τ, gate half-open") is directly
repaired; the residual gap to oracle is per-input overlap (AUC 0.81), not thresholding.

### 4.2 Yi-1.5-9B-Chat, RULER 4K — real compression instead of full-KV fallback

Gate-open fractions: qa_1 0.78 → 1.00, qa_2 0.66 → 1.00, **vt 0.10 → 1.00, fwe 0.00 → 1.00**,
multivalue 0.00 → 0.02, MK3 0.00 → 0.34.

4-task tab:matrix suite (N = 200):

| budget | plain | fixed-τ gated | **z-gated** | oracle | kept-KV fixed-τ → z |
|---:|---:|---:|---:|---:|---|
| 0.5 | 0.665 | 0.865 | 0.800 | 0.870 | 0.890 → **0.583** |
| 0.25 | 0.570 | 0.865 | 0.715 | 0.875 | 0.835 → **0.374** |
| 0.125 | 0.420 | 0.855 | 0.570 | 0.860 | 0.807 → **0.269** |
| 0.0625 | 0.200 | 0.830 | 0.350 | 0.830 | 0.794 → **0.217** |
| **grand-mean Δ vs plain** | | **+0.390** | **+0.145** | +0.395 | |

6-task pooled Δ: +0.323 → +0.156; kept-KV at b = 0.0625: 0.759 → 0.319.

So: **yes, the z-gate makes Yi actually compress on the dilution-labeled tasks** (vt/fwe open
on every input, ~3.5× less KV kept at aggressive budgets) — and it still beats plain SnapKV at
every budget. But it gives back most of the fixed-τ accuracy gain. That give-back must be read
correctly:

### 4.3 Why Yi's +0.390 was never a transferable target

The paper's +0.390 on Yi was earned almost entirely by the gate *staying closed everywhere*
(77/300 open; kept-KV 0.76–0.89 ≈ no compression) while plain SnapKV collapsed. On Yi at 4K the
"dilution-prone" labels are behaviorally wrong for SnapKV: vt (plain mean 0.595 vs full-KV
0.890), fwe (0.465 vs 0.880) and multivalue (0.330 vs 0.700) are all eviction-*fragile*; only
qa_1/qa_2 are truly compressible (plain within 1.5pp of full-KV). Yi's low raw drops on
vt/fwe/mv were therefore *behaviorally correct per-input predictions*, and fixed τ = 0.07 was
"right for the wrong reason" — it happened to sit above almost all of Yi's distribution.
A gate that opens the dilution-labeled tasks (what "correct transfer" means at task level)
*necessarily* pays vt/fwe's fragility cost; even the per-input oracle only reaches +0.395 by
keeping 43–66% of KV. The z-gate is the honest operating point: gated > plain at every budget
with genuine compression, and the θ_z dial exposes the whole trade-off curve:

| θ_z | Yi: matrix Δ / kept@0.0625 / MK3-open | Llama: matrix Δ / kept@0.0625 / MK3-open |
|---:|---|---|
| −0.80 | +0.083 / 0.275 / 0.62 | +0.144 / 0.300 / 0.42 |
| **−0.69** | **+0.145 / 0.319 / 0.34** | **+0.174 / 0.331 / 0.30** |
| −0.50 | +0.200 / 0.366 / 0.06 | +0.209 / 0.372 / 0.16 |
| −0.30 | +0.240 / 0.406 / 0.02 | +0.267 / 0.481 / 0.04 |
| 0.00 | +0.274 / 0.478 / 0.00 | +0.320 / 0.597 / 0.00 |

Δ increases monotonically in θ_z on both held-out models (more conservative = closer to
full-KV fallback); any θ_z in [−0.7, −0.3] is a defensible operating point, and the ranking of
variants does not depend on the choice.

### 4.4 Pilot-size robustness (deployability check)

Bootstrap the unlabeled pilot used for μ, σ (500 resamples), gate at θ_z = −0.69:

| model | pilot n | open(MK3) | open(rest) |
|---|---:|---|---|
| Yi | 40 | 0.394 ± 0.268 | 0.801 ± 0.010 |
| Yi | 100 | 0.371 ± 0.177 | 0.804 ± 0.001 |
| Llama | 40 | 0.327 ± 0.157 | 0.797 ± 0.027 |
| Llama | 100 | 0.296 ± 0.078 | 0.796 ± 0.012 |

The gate on non-MK3 inputs is essentially pilot-independent at n = 40. MK3 protection is
noisier (its z-scores cluster near θ) — at the paper's 40-input pilot the MK3 open-fraction
varies ±0.16–0.27; n ≈ 100 halves this. Recommend stating n = 100 for the unlabeled recipe, or
keeping the existing labeled 40-input midpoint recipe when a known capacity-bound anchor task
is available (the two recipes agree on Qwen/Mistral, §3.2).

## 5. Verdict

1. **Which variant:** per-model pooled **z-score of the raw early-minus-late drop** (variant b),
   single global threshold θ_z = −0.69 fit on Qwen/Mistral cells only. It is deployable (one
   unlabeled ~100-input pilot per model — no labels, no training, no backward pass), achieves
   7/7 task-level separation of MK3 from qa_1/qa_2/vt/fwe (raw τ: 4/7), pooled cross-cell AUC
   0.965 (0.990 excluding multivalue), and is exactly the recipe the paper already pays for,
   minus the label requirement. Quantile normalization (d) is a close second (7/7, weaker MK3
   protection at the calibrated threshold); min-max (c) is dominated; task-max (a) fails even
   as a diagnostic; shape variants (e) actively destroy the signal.
2. **Does it fix transfer?** On Llama-3.1-8B: yes, unambiguously — gated Δ +0.109 → **+0.174**
   (6-task +0.089 → +0.134), with MK3 protection restored (open 0.56 → 0.30). On Yi-1.5-9B: it
   fixes what was actually broken — the gate now compresses (kept-KV 0.79 → 0.22–0.32 at
   aggressive budgets, vt/fwe/qa open ≈ 1.0) and still beats plain at every budget (+0.145
   matrix Δ) — but it cannot and should not reproduce +0.390, which was a full-KV-fallback
   artifact on a model where even the dilution-labeled tasks are eviction-fragile (oracle needs
   43–66% kept-KV to reach +0.395).
3. **Text claim it supports:** "The early-to-late head-agreement drop is an
   architecture-invariant *ordering* of eviction risk; only its location and scale are
   model-specific. Standardizing D with an unlabeled ~100-input pilot (z-score) lets a single
   global threshold θ_z = −0.69, calibrated on Qwen/Mistral, transfer zero-shot to two held-out
   Llama-family models: task-level separation of the capacity-bound anchor (NIAH-MK3) from all
   four dilution-prone tasks in 7/7 model-cells, gated-eviction gains at every budget on both
   held-out models, and on Llama-3.1-8B a +6.5pp improvement over the fixed-τ gate at
   comparable compression."
   The claim must **not** say the z-gate preserves Yi's +0.390 (it trades accuracy for real
   compression there, +0.145), and must scope niah_multivalue out of the invariant task-level
   ordering (its drop falls below MK3 on both Llama-family models — behaviorally the right
   call there, but a task-label misclassification).

Caveats: (i) per-input MK3-vs-rest overlap on Llama-family models is intrinsic (AUC ≈ 0.80
either normalization or none; 0.97–1.00 excluding multivalue) — 30–34% of MK3 inputs open at
θ_z = −0.69, costing ~0.14–0.25 of MK3 accuracy vs full-KV; (ii) the Llama gated jsonl's stored
run-time drops differ from the probe-file recomputation by up to 7.4e−3 (17/300 τ-decisions
flip) — all numbers here use the recomputed convention consistently; (iii) Yi full-KV outcomes
are recovered from originally-gate-closed rows (223/300 inputs); every input the z-gate closes
is among them, so both re-evaluations are exact.
