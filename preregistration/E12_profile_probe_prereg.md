# Pre-registration: E12 — learned probe over the per-layer agreement profile a_ℓ

**Written before the probe was fit.** Hash this file and record the digest in
the run log / analysis output before fitting. Precedent: 32K test, E8, E9, E11.

Date: 2026-08-26
Round: new-exp-page-kv-r2
Answers: reviewer **Q2 / W4** (ICLR review) and the paper's own stated open
problem (`main.tex:382`, Limitations "Threshold portability"):

> "Why not fit a minimal classifier (logistic regression, 2–3 parameters) over
> the per-layer agreement profile a_ℓ, calibrated with the same unlabeled pilot
> used for z-scoring? A.3.3 tests three hand-crafted profile statistics and
> reports the Llama profile carries signal with inverted sign — this seems to be
> exactly the case a learned combination would repair."

The paper's D collapses the profile a_ℓ to its endpoints (early − late). On
Llama-3.1-8B the profile is non-monotone with a mid-stack spike (documented in
`page-kv/.../layer_profile.py`), so the endpoint contrast inverts and per-input
AUC sits at ≈ 0.80 under the z-scored variant. E12 tests whether a tiny learned
probe over the *interior* of the profile raises that AUC.

---

## Data (frozen)

- **Profiles**: the full per-layer agreement sequence a_ℓ, field
  `head_agreement_per_layer`, from the released dumps
  `drops_llama31_8b_4k.jsonl` (Llama, L = 32) and `drops_qwen15b_4k_n100.jsonl`
  (Qwen2.5-1.5B fitting reference, L = 28). Six RULER tasks × 50 inputs each on
  Llama.
- **Per-input label = the paper's own AUC label, task identity.** The paper's
  per-input AUC (`main.tex:244`, `heldout_ablation.py`) is defined as the
  separability of **NIAH-MK3 inputs (positive) from the dilution-prone-task pool
  (negative: vt, fwe, qa_1, niah_multivalue)**, scored by the gate signal. The
  label is therefore `task == niah_multikey_3`, which is present **in the profile
  dump itself** (`task` field). No cross-file join and no eviction log are
  needed: features and label both come from `drops_*.jsonl`.

  > **Bug caught in audit (was: eviction-derived label).** An earlier draft
  > defined the label as "full-KV-correct AND destroyed by eviction at ≥1
  > budget," joined from the eviction logs. Running it showed that label is
  > degenerate for this purpose: it fires on 65/100 fit-cell inputs and only
  > 2/50 eval-cell inputs (AUC on 2 positives is noise), and it is **not** the
  > construction behind the paper's ≈0.80 figure. The frozen definition above is
  > the paper's actual label, making the probe directly comparable to the
  > z-scored-D baseline the reviewer cites.

- **Baseline reproduction guard (frozen):** the `drop_D` AUC recomputed by this
  script (endpoint drop from the profile, MK3-vs-dilution) must reproduce the
  paper's reported per-input AUC for the fitting and held-out cells
  (`heldout_ablation.py`: Qwen2.5-1.5B 1.000; Llama-arch ≈ 0.80) to within the
  bf16 residual; a mismatch aborts, since it means the label/feature construction
  does not match the paper's.

## Probe (frozen)

- **Normalization**: z-score each profile position using mean/std computed on an
  **unlabeled ~100-input pilot** (the same pilot construction the paper's
  z-scored τ uses). No test label touches the normalization.
- **Features (≤ 3, fixed now)**: over the z-normalized interior profile
  (layers excluding the first and last bin that D already sees):
  (i) `min a_ℓ` over interior layers, (ii) normalized argmin depth
  (layer index of the interior minimum / L), (iii) mid-band mean (middle third).
  These are exactly the interior structure D discards.
- **Model**: logistic regression with an intercept, ≤ 3 weights, fit by
  Newton–Raphson / gradient descent in **pure numpy** (the repro env
  `page-repro` has numpy 2.5.1 but not sklearn/scipy; a dependency-free fit also
  keeps the script self-contained).
- **Fit / evaluation split**: fit on the Qwen2.5-1.5B fitting cell (the cell the
  paper fits everything on); evaluate per-input AUC (MK3-vs-dilution) on the
  **held-out Llama-3.1-8B** cell. Report Qwen in-sample AUC too, but the headline
  is the Llama held-out number.
- **Baselines on identical held-out inputs**: (i) z-scored D — endpoint drop
  standardized on the eval cell's own unlabeled pilot (the paper's current best,
  AUC ≈ 0.80 target to beat); (ii) raw D. Both scored MK3-vs-dilution, higher D
  ⇒ less MK3-like, so the AUC uses −D.

  > Note on the fit cell: on Qwen2.5-1.5B the paper's D already separates
  > MK3-vs-dilution at AUC 1.000, so the probe cannot *improve* in-sample there;
  > the fit cell exists to learn interior-profile weights, and the test of value
  > is purely the **held-out Llama** number. If the released Llama profile dump
  > has too few MK3 inputs for a stable held-out AUC (it has 50 MK3 vs ~200
  > dilution — adequate), the id-tagged re-dump below adds more.

## Predictions (frozen)

- **Success criterion (pre-committed):** the probe **closes the gap** iff its
  held-out Llama AUC ≥ (z-scored D AUC) + 0.05. A result in [z-scored D,
  +0.05) is "partial"; ≤ z-scored D is "no improvement — the endpoint statistic
  is not the bottleneck."
- **P1.** Probe held-out AUC > raw-D AUC (interior structure carries signal the
  endpoint inverts on Llama).
- **P2.** The learned interior-minimum feature (i) receives non-negligible
  weight (|w| above the intercept-only null), i.e. the mid-stack spike is what
  the endpoint contrast misses.

**Honesty stance.** All three outcomes (close / partial / no improvement) are
reported as-is. Features are **not** re-selected after seeing the held-out AUC;
the feature set above is frozen by this hash. A "no improvement" result is a
finding about *where* the Llama failure lives, not a failed experiment.

## Compute

Core result is **offline** (released dumps + eviction logs, CPU-only numpy fit).
An optional id-tagged re-dump of profiles (adding an explicit `id` field via a
round-local copy of `per_layer_agreement_probe.py`) hardens the join but is not
required for the headline; if run it is prefill-only, minutes per 100 inputs.

## Addendum (measurement notes, appendable without invalidating the hash)

Frozen portion ends at "## Addendum"; the digest in
`E12_profile_probe_prereg.sha256` covers only the frozen portion.

---

## Addendum (post-run measurement notes; does not alter the frozen hash)

Recorded 2026-08-26 after the offline run on released logs.

- **Guard target corrected during audit.** An earlier draft set the raw-D AUC
  guard on Llama to 0.80. That is wrong: 0.80 is the *z-scored threshold*
  figure; raw drop_D on the Llama held-out cell is **0.741** (`main.tex:244`).
  Target set to 0.741. Guard now passes: fit AUC 1.000, eval raw-D 0.741.
- **z-scored D AUC == raw D AUC (0.741).** z-scoring a single scalar is a
  monotonic transform, so it cannot change that scalar's AUC. The paper's ~0.80
  therefore describes calibrated *threshold* transfer, not per-input AUC on this
  cell. Recorded so no one re-derives this as a discrepancy.
- **Result: NO IMPROVEMENT (prereg outcome (c)).** Probe held-out AUC 0.571 <
  raw/z-scored D 0.741. A Qwen-fit interior-profile probe transfers *worse* than
  the endpoint statistic to Llama — the interior structure that breaks D also
  differs across architectures, so a naively-transferred learned combination
  does not repair the gap. This is the informative, honest answer to Q2: the
  fix is not "just fit a probe on the fitting cell." A probe fit *on Llama's own
  pilot* (in-architecture) is the remaining untested variant; flagged as the
  next step, not claimed here.
