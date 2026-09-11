# Results index

This directory holds every experiment behind the paper: the originally
released per-input logs plus three later rounds of verification, robustness,
and generalization checks. Every summary `.md` here is regenerated from those
logs by a script in `../scripts/`, and every script asserts its own numbers
so it fails loudly if they stop reproducing.

## `.jsonl` schema

Every `*.jsonl` file here is one row per (input, budget) pair. The core
fields, written by `../scripts/gated_eviction.py` and shared by most runners:

| field | meaning |
|---|---|
| `id` | input index within its task |
| `task` | RULER/LongBench/AgentLongBench task name |
| `budget` | kept-KV fraction tested (1.0 = full cache) |
| `T` | prompt length in tokens |
| `drop` | the head-agreement drop statistic `D` |
| `gate_open` | whether `D >= tau` (eviction runs) at this row's tau |
| `score_policy` | which base evictor scored this row (snapkv/h2o/streamingllm/pyramidkv/manifoldkv/...) |
| `n_kept_plain`, `n_kept_gated` | tokens actually retained, plain vs. gated arm |
| `gold` | reference answer(s) |
| `pred_plain`, `pred_gated` | generated text, plain vs. gated arm (truncated to 200 chars) |
| `correct_plain`, `correct_gated` | exact/substring-match correctness, plain vs. gated arm |

A few runners extend this with allocation-specific fields — e.g.
`adakv_matrix.py` replaces `correct_plain`/`correct_gated` with
`correct_plain_shared`/`correct_plain_adakv`/`correct_gated_shared`/
`correct_gated_adakv` to compare the shared-mask and per-head allocations
side by side. Check the `rec = {...}` construction in the specific runner
(named in the per-file tables below) for the exact fields it writes.

Regenerate the core zero-GPU set:

```bash
cd ..            # experiments/
./run_all.sh     # 11 steps, exits non-zero on any CHECK failure
```

Regenerate the budget/batching/baseline-fidelity set:

```bash
cd ../scripts
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate page-repro
python matrix_budget_restricted.py
python batching_decay.py
python h2o_degeneracy_audit.py
python scaling_slope_fit.py
python adakv_analysis.py          # needs the per-head Ada-KV GPU run
```

Regenerate the generalization/robustness set (zero-GPU on released logs):

```bash
cd ../scripts
python prevalence_survey.py
python profile_probe_ablation.py
python dynamickv_headtohead_analysis.py
```

**Compute requirements.** GPU memory is documented per cell where it drove a
design decision (e.g. Qwen2.5-14B's two-card sharding, below). Wall-clock
runtime per GPU cell is **TODO: verify** — it was not recorded in a form this
checkout preserves; expect single-model RULER-4K cells (100 examples, a
handful of budgets) to be the fastest, Qwen2.5-14B and 16K/32K-context cells
the slowest, but no specific hour/minute figure should be assumed without
timing it yourself.

Paths resolve relative to this checkout via `../scripts/paths.py`, so the
commands above need no env setup. `PAGE_RESULTS` points at the per-input logs,
`PAGE_DATA` at GPU outputs from the later rounds, `PAGE_OUT` at where
summaries are written; all three default to this directory. GPU runners are
in `../gpu/`, run logs in `../logs/` (not published; see `.gitignore`), and
every frozen pre-registration in `../../preregistration/`.

`adakv_analysis.py` exits non-zero by design: pre-registered prediction P1 is
genuinely violated on Mistral. That is the recorded finding, not a broken run.

---

## Core partition & gate validation

| file | what it settles | script |
|---|---|---|
| `seed_variance.md` | Seed SD is 0.005–0.011 against a +22.9pp headline, so the four single-seed headline cells are not seed-fragile. Per-cell SD for the headline matrix. | `seed_variance.py` |
| `heldout_ablation.md` | `D` is the only one of six signals that ever separates, on 3 of 4 cells. AUC 1.000 is fitting-cell only (held out: 0.899, 0.978, 0.741). The ordering **inverts** on Llama-3.1-8B. | `heldout_ablation.py` |
| `head_pair_subsample.md` | The task ordering survives on **1 of 66 head pairs**, 20/20 draws at every fraction. Answers the `O(L·H²·k)` deployment cost objection. | `head_pair_subsample.py` |
| `kappa_theorem2.md` | Measured κ ≈ 3.3 against the ≈10.1 the bound needs, so Theorem 2 and Corollary 2 are vacuous at the paper's own (H, L, T). A self-reported negative. | `measure_kappa.py` |
| `wrapper_validation.md` | Which memory workarounds are trustworthy. v1 and v3 faithful; **v2 shifted `D` by 33%** and is used by no reported cell. | (2×2 runs) |

**A DynamicKV comparison against the raw runner is deliberately absent from
this section.** The reimplementation's adaptive per-layer budget scored
*worse* than the uniform-budget control arm (−0.040 mean, −0.130 at
b=0.125). With no public reference implementation, that is evidence about
this codebase rather than about the published method, so it is not reported
here — see the DynamicKV head-to-head under Generalization below for the
version that does get reported (gate vs. adaptive allocation, not a bare
reimplementation comparison).

### Zero-GPU results

| file | what it settles |
|---|---|
| `harm_rate.md` | Fixed-budget harm rate, the replacement for a vacuous `r = -1` claim. Includes the provenance note on two similarly named 16K logs. |
| `budget1_identity_audit.md` | `r(x) ≥ 0` is an identity, audited over 5,120 rows on 20 cells. |
| `per_task_delta_matrix.md` | The matrix is one measurement replicated 16×: grand mean +22.9pp, +4.5pp excluding NIAH-MK3, exactly 0.000 in 8 of 16 cells. |
| `task_label_oracle.md` | A cheating task-label oracle reaches +20.1pp against PAGE's +22.9pp; identical in half the cells, PAGE ahead in 5. |
| `mixture_sweep.md` | Δ as a function of the capacity-bound share of the workload. |
| `layer_profile.md`, `layer_subsample.md` | The per-layer agreement profile, and how many layers `D` needs. |

### Figures

| file | note |
|---|---|
| `fig1_panel3.pdf` | Figure 1 panel 3, drawn from measured curves. |
| `fig1_heatmaps.pdf` | The same panel's heatmaps as top-k **set membership**, which is what `D`'s Jaccard actually measures. Asserts the rendered example matches its caption. |
| `layer_profile.pdf` | The per-layer profile for Qwen vs Llama; the non-monotone Llama profile is the mechanism behind that family's transfer failure. |
| `mixture_sweep.pdf` | Δ vs capacity-bound workload share. |

### Standing caveats

* All GPU runs used a fresh env matching the reproducibility statement
  (Python 3.13.14 / torch 2.11.0 / transformers 5.9.0). See `repro_gate.md`:
  **HOST_A (A100) reproduces the released log exactly**, 800/800 on every
  hard field and `max|Δdrop| = 0.000e+00`, and every reported run lives there.
  On HOST_B (Ada) generations drift on ~8% of inputs while gate decisions,
  kept-counts and τ crossings stay 100% identical.
* Seed replicates within a cell always share a host, so the spread is the input
  draw and not the hardware.
* All 15 seed runs (5 cells × 3 seeds) are complete; `seed_variance.py` refuses
  files short of 3,600 rows rather than reporting a partial one.

---

## Budget, batching & baseline-fidelity audits

| file | what it settles | script |
|---|---|---|
| `matrix_budget_restricted.md` | **Aggressive-budget restriction.** Restricting to the >= 4x budgets **raises** the mean. Raw +0.2286 -> +0.2584, but the cells do not share a budget grid; on a **matched grid** the effect is **+0.0167** (+0.2402 -> +0.2569). Quote the matched-grid number. | `matrix_budget_restricted.py` |
| `batching_decay.md` | **Batch-size compression decay.** Expected compression vs batch size under static provisioning: **2.91x at B=1 falling to 1.01x by B=16**. (Mean of per-cell compressions; the model is nonlinear in p_open, so averaging inputs first is wrong.) | `batching_decay.py` |
| `h2o_degeneracy_audit.md` | **H2O/SnapKV column duplication.** The Qwen2.5-14B H2O column agrees with SnapKV on **600/600 evicting rows**; the other three sit at **71-73%**. An undisclosed duplicate in a headline table. (b=1.0 excluded: both arms are the full cache there.) | `h2o_degeneracy_audit.py` |
| `scaling_slope_fit.md` | **Scaling-figure slope, actually fitted.** The figure's asserted slope ~0.4 is unfitted; through-origin OLS is **0.3432** (dilution-only) or 0.2111 (with MK3). FWE-4K at 0.042 is a clear counterexample to the [0.25, 0.55] band; 16K VT at 0.556 is a boundary case its 2-dp inputs cannot resolve. | `scaling_slope_fit.py` |
| `adakv_matrix.md` | **Per-head Ada-KV, strongest form of the baseline.** Per-head Ada-KV plain arm costs **1.1pp** (+0.2575 -> +0.2467 at b<=0.25) across **all 4 models**. P1 violated on **Mistral only** (4 of 32 cell-budget points); Qwen 1.5B/3B/14B all 0/8. Amendment recorded in the pre-registration. | `adakv_analysis.py` |

### What each one costs the paper, honestly

Some of these help and some hurt. All are reported.

* **The budget-restriction check is a win.** Restricting to aggressive
  budgets was suspected to deflate the headline. It raises it, so the
  reported +22.9pp is conservative with respect to the budgets the caption
  endorses.
* **The H2O degeneracy audit is the good and bad halves of one measurement.**
  Bad: the matrix has 15 independent cells, not 16, and does not say so.
  Good: `drop` is identical to 1e-9 across scorers on 100% of rows in every
  model, which is the method-agnosticism claim measured rather than
  asserted.
* **The batching-decay check is unflattering by design.** Under static batch
  provisioning the memory benefit is gone by batch 16. The paper should
  state the batch-1/offline scope in the abstract rather than leaving the
  mechanism unquantified. Note the modelling scope: continuous batching with
  paged allocation does not behave this way, so the table is an upper bound
  on the damage.
* **The scaling-slope fit weakens a claim.** An honest fit does not support
  "clusters in [0.25, 0.55]". But three mutually inconsistent numbers for
  one slope is worse, and it is a two-minute recomputation to check.
* **Per-head Ada-KV narrows the central claim.** The capacity-bound class is
  near-tie *surface-form* distractor retrieval, not a general class. `qa_1`
  and `qa_2`, with 19 and 29 real hard-negative documents, have the two
  **highest** D values in the whole suite. Reported because it is true, and
  it is more defensible than the current wording.

### Per-head Ada-KV pre-registration

`../../preregistration/adakv_perhead_prereg.md`, sha256 of the frozen predictions
section `c112a20265464a4f...`, hashed and recorded in the run log **before**
the first GPU job started. `paths.verify_prereg()` re-checks it at launch and
at analysis time, hashing only the predictions so the setup addendum can be
appended without invalidating the commitment.

Predictions P1-P5 and the falsification table are in that file. The analysis
script evaluates each explicitly and prints VIOLATED where it applies,
including outcomes that damage the paper.

### Standing caveats

* All runs use conda env `page-repro` (Python 3.13.14 / torch 2.11.0+cu130 /
  transformers 5.9.0) on **HOST_A** (4x A100-SXM4-80GB), the host that
  reproduces the released logs exactly.
* **Gate-signal fidelity, measured per model.** On Qwen2.5-1.5B the per-head
  Ada-KV runner reproduces released `D` at **0.00e+00** on 5/5 inputs with
  matching `T`. On Qwen2.5-14B it reproduces only to ~1e-4. Three controls
  localised that gap to pre-existing environment drift rather than anything
  this runner introduces: one-pass vs two-pass both land ~1e-4 (one-pass
  closer); sharded vs single-card are bit-identical to each other; and the
  attention patch itself is exactly neutral. At ~1e-4 against tau = 0.07 it
  cannot flip a gate call.
* **A reimplementation hazard, found the hard way.** Rewriting `build_prompt`
  rather than mirroring `gated_eviction.py` shifted `D` by 4e-4 through a
  single `\n\n` vs `\n` before the answer prefix. The runner now mirrors the
  canonical prompt exactly. Any new runner must be diffed against released `D`
  before its numbers are believed.
* **Qwen2.5-14B needs two cards.** At 4K it materialises 64.4 GB of attention
  tensors (48 layers x 40 heads) plus ~28 GB of weights, so it OOMs on one
  80 GB card in one-pass mode. The runner shards it with `device_map=auto`,
  which keeps it on the same one-pass path as the other three cells.
  Two-pass was rejected as the fix precisely because it changes what the
  scorer sees, which is exactly the H2O/SnapKV degeneracy documented above.
* Partial run files are **refused, not summarised**. `adakv_analysis.py`
  reports a cell as incomplete rather than averaging a short file.
* **Double-BOS tokenization.** The runner initially omitted
  `add_special_tokens=False` (which `gated_eviction.py:538` uses). Mistral and
  Llama chat templates already emit BOS, so a second was prepended: on Mistral
  that drifted `D` up to 1.13e-02 and **flipped 18 of 400 gate calls**. Qwen
  templates emit no BOS, so those cells were unaffected and hid the bug until
  it was caught. Fixed; post-fix Mistral reproduces the released log exactly.
  Any new architecture must be diffed against its released `T` and `D` first.
* **New model families must be added to `patch_families()`.** Without the
  patch the per-head mask silently does not apply and both arms become
  identical, which looks like a null result rather than an error.

---

## Generalization & robustness

Beyond the core matrix, these four experiments test whether the gate's
claims hold up under conditions the headline results don't directly probe:
a different architecture, a different (realistic) workload, a richer
predictor, and a stronger adaptive baseline.

| file | what it settles | script |
|---|---|---|
| `llama_architecture_bias_control.md` | **Architecture bias control** (corroborated by 14B at 0/8). Llama-3.1-8B (same L=32,Q=32,KV=8 as Mistral, different family) violates P1 at **0/8** budgets and shows **+0.140** on MK3 where Mistral shows **-0.440**. Refutes the architectural reading of the Mistral anomaly. | manual, from `adakv_4k_llama31.jsonl` |
| `capacity_bound_class_generalization.md` | **Does the capacity-bound class generalize beyond one synthetic task?** `cwe` is not capacity-bound on any of three models (gate open on 0/300); `niah_multiquery` is capacity-bound on 2 of 3. Narrows the class to near-tie surface-form retrieval, not distractor count in general. | manual analysis over `capbound_4k_*.jsonl` |
| `prevalence_survey.md` | **How common is the capacity-bound class in realistic traffic?** Pooled share **f = 3.65% [2.3%, 5.7%]** at matched b≤0.125 over 9 Qwen2.5-14B LongBench subtasks; `lcc` capacity-bound, `repobench-p` mixed (two realistic code members). AgentLongBench (32K+, out of fitting range) reported separately. | `prevalence_survey.py` |
| `profile_probe_ablation.md` | **Does a richer per-layer feature out-predict the simple endpoint statistic D?** Raw-D AUC on Llama is 0.741 (reproduces the paper exactly); a Qwen-fit interior-profile logistic probe does **NOT** close the gap (0.571 < 0.741). The endpoint statistic is not the bottleneck; the Llama transfer failure lives elsewhere. | `profile_probe_ablation.py` |
| `dynamickv_headtohead_analysis.md` | **Does adaptive per-layer budget allocation alone protect the capacity-bound task?** No: DynamicKV-style adaptive allocation collapses MK3 to 0.000 at b≤0.125, identically to a uniform-budget control; the gate recovers +1.000. Allocation ≠ admission — the gate's per-input decision is the necessary addition, not just smarter allocation. | `dynamickv_headtohead_analysis.py` |

### What each one costs the paper, honestly

* **The architecture-bias control stopped a false generalisation.** The
  Mistral P1 anomaly looked architectural (8 KV-heads). Llama-3.1-8B at
  identical head geometry shows the opposite sign, so the generalisation is
  wrong and was not written.
* **The class-generalization probe narrows the central claim.** The
  capacity-bound class is near-tie *surface-form* distractor retrieval, not
  a general "many distractors" class — `qa_1`/`qa_2` have the highest D
  values in the suite despite having the most distractors.
* **The prevalence survey is a genuine bound, not a comfortable one.** At
  ~3.7% of realistic traffic, the class the gate protects is a small
  minority of inputs — the paper's framing should say "concentrated," not
  imply it is common.
* **The profile-probe ablation is an honest negative result.** A more
  expressive predictor was a natural next step to try, and it did not help.
  The Llama transfer problem is not fixed by a richer feature over the same
  attention profile.
* **The DynamicKV head-to-head answers the strongest baseline objection.**
  "Adaptive allocation already does what PAGE does" is a real alternative
  hypothesis; measured against a working adaptive-budget baseline (not just
  asserted), it does not hold.

### Prevalence-survey / profile-probe / DynamicKV pre-registrations

`../../preregistration/prevalence_survey_prereg.md`,
`../../preregistration/profile_probe_ablation_prereg.md`, and
`../../preregistration/dynamickv_headtohead_prereg.md` — same sha256-hash
convention as the per-head Ada-KV pre-registration above: predictions frozen
before any GPU job started, verified by the launch script before it claims a
card.

### Standing caveats

* The prevalence survey pools 9 of 12 target subtasks from released,
  zero-GPU logs; the remaining 3 (2wikimqa, musique, gov_report) and the
  AgentLongBench agentic slice need `experiments/gpu/run_prevalence_gaps.sh`.
  AgentLongBench's accuracy fields are not a reliable signal (short-integer
  answers make substring matching near-uninformative), so only its
  gate-closure fields are used, and it is never pooled into the in-range f.
* The profile-probe ablation fits on Qwen2.5-1.5B and evaluates held-out on
  Llama-3.1-8B; `experiments/gpu/run_profile_probe.sh` can produce an
  id-tagged re-dump to add a Llama-own-pilot in-architecture variant, not
  yet run.
* The DynamicKV head-to-head's released cell is Qwen3-4B only;
  `experiments/gpu/run_dynamickv_headtohead_analysis.sh` broadens it to
  Qwen2.5-3B and Mistral-7B via the same unchanged runner.

---

## Reproducibility

| file | note |
|---|---|
| `repro_gate.md` | HOST_A (A100) reproduces the released log exactly on every hard field; HOST_B (Ada) matches on gate decisions/kept-counts/τ-crossings but drifts ~8% on decoded text (expected under greedy decoding on different hardware). |
| `wrapper_validation.md` | Which memory-saving wrapper variants are faithful to the stock (non-wrapped) attention path — needed because full attention materialization for held-out cells (14B, Llama-8B, Mistral-16K) does not fit in memory without one. |
| `control_2x2.md` | The 2×2 (code-path × hardware) control that separates a wrapper-induced statistic shift from a hardware-induced one. |
