# PAGE revision rounds — verified results

Two revision rounds (**R1** and **R2**) were merged into this directory
alongside the originally released per-input logs. Every summary here is
regenerated from those logs by a script in `../scripts/`, and every script
asserts its own numbers so it fails loudly if they stop reproducing.

Regenerate the R1 zero-GPU set:

```bash
cd ..            # experiments/
./run_all.sh     # 11 steps, exits non-zero on any CHECK failure
```

Regenerate the R2 set:

```bash
cd ../scripts
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate page-repro
python matrix_budget_restricted.py
python batching_decay.py
python h2o_degeneracy_audit.py
python scaling_slope_fit.py
python adakv_analysis.py          # needs the E8 GPU run
```

Paths resolve relative to this checkout via `../scripts/paths.py`, so the
commands above need no env setup. `PAGE_RESULTS` points at the per-input logs,
`PAGE_DATA` at the rounds' GPU outputs, `PAGE_OUT` at where summaries are
written; all three default to this directory. GPU runners are in `../gpu/`,
run logs in `../logs/`, and R2's frozen pre-registrations in
`../../preregistration/`.

`adakv_analysis.py` exits non-zero by design: pre-registered prediction P1 is
genuinely violated on Mistral. That is the recorded finding, not a broken run.

---

# Round 1

## GPU results

| file | what it settles | script |
|---|---|---|
| `seed_variance.md` | **P3-1.** Seed SD is 0.005–0.011 against a +22.9pp headline, so the four single-seed headline cells are not seed-fragile. Gives per-cell SD for `tab:matrix`. | `seed_variance.py` |
| `heldout_ablation.md` | **P3-2.** `D` is the only one of six signals that ever separates, on 3 of 4 cells. AUC 1.000 is fitting-cell only (held out: 0.899, 0.978, 0.741). The ordering **inverts** on Llama-3.1-8B. | `heldout_ablation.py` |
| `head_pair_subsample.md` | **P3-3.** The task ordering survives on **1 of 66 head pairs**, 20/20 draws at every fraction. Answers the `O(L·H²·k)` deployment objection. | `head_pair_subsample.py` |
| `kappa_theorem2.md` | **T-6.** Measured κ ≈ 3.3 against the ≈10.1 the bound needs, so Theorem 2 and Corollary 2 are vacuous at the paper's own (H, L, T). A self-reported negative. | `measure_kappa.py` |
| `wrapper_validation.md` | Which memory workarounds are trustworthy. v1 and v3 faithful; **v2 shifted `D` by 33%** and is used by no reported cell. | (2×2 runs) |

**P3-4 (DynamicKV) is deliberately absent.** The reimplementation's adaptive
per-layer budget scored *worse* than the uniform-budget control arm (−0.040
mean, −0.130 at b=0.125). With no public reference implementation, that is
evidence about our code rather than about their method, so it is not reported.
The revision plan's fallback applies: argue the per-input-gate vs
per-task-budget distinction in prose and concede the gap.

## Zero-GPU results

| file | what it settles |
|---|---|
| `harm_rate.md` | Fixed-budget harm rate, the replacement for the vacuous `r = -1` claim. Includes the provenance note on the two similarly named 16K logs. |
| `budget1_identity_audit.md` | `r(x) ≥ 0` is an identity, audited over 5,120 rows on 20 cells. |
| `per_task_delta_matrix.md` | The matrix is one measurement replicated 16×: grand mean +22.9pp, +4.5pp excluding NIAH-MK3, exactly 0.000 in 8 of 16 cells. |
| `task_label_oracle.md` | A cheating task-label oracle reaches +20.1pp against PAGE's +22.9pp; identical in half the cells, PAGE ahead in 5. |
| `mixture_sweep.md` | Δ as a function of the capacity-bound share of the workload. |
| `layer_profile.md`, `layer_subsample.md` | The `a_ℓ` profile behind (A3), and how many layers `D` needs. |

## Figures

| file | note |
|---|---|
| `fig1_panel3.pdf` | Figure 1 panel 3, drawn from measured curves. The hand-drawn version had the dilution-prone case inverted, which would have implied `D < 0` while the panel was labelled "large `D`". |
| `fig1_heatmaps.pdf` | The same panel's heatmaps as top-k **set membership**, which is what `D`'s Jaccard actually measures. Asserts the rendered example matches its caption. |
| `layer_profile.pdf` | `a_ℓ` for Qwen vs Llama; the non-monotone Llama profile is the mechanism behind that family's transfer failure. |
| `mixture_sweep.pdf` | Δ vs capacity-bound workload share. |

## Standing caveats

* All GPU runs used a fresh env matching the reproducibility statement
  (Python 3.13.14 / torch 2.11.0 / transformers 5.9.0). See `repro_gate.md`:
  **jagannath (A100) reproduces the released log exactly**, 800/800 on every
  hard field and `max|Δdrop| = 0.000e+00`, and every reported run lives there.
  On sateri (Ada) generations drift on ~8% of inputs while gate decisions,
  kept-counts and τ crossings stay 100% identical.
* Seed replicates within a cell always share a host, so the spread is the input
  draw and not the hardware.
* All 15 seed runs (5 cells × 3 seeds) are complete; `seed_variance.py` refuses
  files short of 3,600 rows rather than reporting a partial one.

---

# Round 2

## Scope of this round

R2 runs the items the R1 plan listed under "Deliberately not doing", plus the
four zero-GPU reanalyses that were cheap enough to bank alongside them. The
DynamicKV head-to-head and the 32K throughput benchmark were **not** run: both
were excluded for reasons compute does not fix (no public reference
implementation; a hard memory wall), and the standing instruction is to
escalate rather than write them up if they reproduce their known failure.

## Results

| file | what it settles | script |
|---|---|---|
| `matrix_budget_restricted.md` | **E1 / W2.** Restricting to the >= 4x budgets **raises** the mean. Raw +0.2286 -> +0.2584, but the cells do not share a budget grid; on a **matched grid** the effect is **+0.0167** (+0.2402 -> +0.2569). Quote the matched-grid number. | `matrix_budget_restricted.py` |
| `batching_decay.md` | **E2 / W3.** Expected compression vs batch size under static provisioning: **2.91x at B=1 falling to 1.01x by B=16**. (Mean of per-cell compressions; the model is nonlinear in p_open, so averaging inputs first is wrong.) | `batching_decay.py` |
| `h2o_degeneracy_audit.md` | **E3 / W9.** The Qwen2.5-14B H2O column agrees with SnapKV on **600/600 evicting rows**; the other three sit at **71-73%**. An undisclosed duplicate in a headline table. (b=1.0 excluded: both arms are the full cache there.) | `h2o_degeneracy_audit.py` |
| `scaling_slope_fit.md` | **E6 / W12.** The figure's asserted slope ~0.4 is unfitted; through-origin OLS is **0.3432** (dilution-only) or 0.2111 (with MK3). FWE-4K at 0.042 is a clear counterexample to the [0.25, 0.55] band; 16K VT at 0.556 is a boundary case its 2-dp inputs cannot resolve. | `scaling_slope_fit.py` |
| `adakv_matrix.md` | **E8 / W2 strongest form.** Per-head Ada-KV plain arm costs **1.1pp** (+0.2575 -> +0.2467 at b<=0.25) across **all 4 models**. P1 violated on **Mistral only** (4 of 32 cell-budget points); Qwen 1.5B/3B/14B all 0/8. Amendment recorded in the prereg. | `adakv_analysis.py` |
| `E10_llama_control.md` | **E10 / bias control** (corroborated by 14B at 0/8). Llama-3.1-8B (same L=32,Q=32,KV=8 as Mistral, different family) violates P1 at **0/8** budgets and shows **+0.140** on MK3 where Mistral shows **-0.440**. Refutes the architectural reading. | manual, from `adakv_4k_llama31.jsonl` |

## What each one costs the paper, honestly

Some of these help and some hurt. All are reported.

* **E1 is a win.** The reviewer assumed restricting to aggressive budgets would
  deflate the headline. It raises it, so the reported +22.9pp is conservative
  with respect to the budgets the caption endorses.
* **E3 is the good and bad halves of one measurement.** Bad: the matrix has 15
  independent cells, not 16, and does not say so. Good: `drop` is identical to
  1e-9 across scorers on 100% of rows in every model, which is the
  method-agnosticism claim measured rather than asserted.
* **E2 is unflattering by design.** Under static batch provisioning the memory
  benefit is gone by batch 16. The paper should state the batch-1/offline scope
  in the abstract rather than leaving the mechanism unquantified in Section 5.
  Note the modelling scope: continuous batching with paged allocation does not
  behave this way, so the table is an upper bound on the damage.
* **E6 weakens a claim.** An honest fit does not support "clusters in
  [0.25, 0.55]". But three mutually inconsistent numbers for one slope is
  worse, and a reviewer recomputes it in two minutes.
* **E9 narrows the central claim.** The capacity-bound class is near-tie
  *surface-form* distractor retrieval, not a general class. `qa_1` and `qa_2`,
  with 19 and 29 real hard-negative documents, have the two **highest** D
  values in the whole suite. Reported because it is true, and it is more
  defensible than the current wording.
* **E10 stopped a false generalisation.** The Mistral P1 anomaly looked
  architectural (8 KV-heads). Llama-3.1-8B at identical head geometry shows the
  opposite sign, so the generalisation is wrong and was not written.

## E8 pre-registration

`../../preregistration/adakv_perhead_prereg.md`, sha256 of the frozen predictions
section `c112a20265464a4f...`, hashed and recorded in the run log **before**
the first GPU job started. `paths.verify_prereg()` re-checks it at launch and
at analysis time, hashing only the predictions so the setup addendum can be
appended without invalidating the commitment.

Predictions P1-P5 and the falsification table are in that file. The analysis
script evaluates each explicitly and prints VIOLATED where it applies,
including outcomes that damage the paper.

## Standing caveats

* All runs use conda env `page-repro` (Python 3.13.14 / torch 2.11.0+cu130 /
  transformers 5.9.0) on **jagannath** (4x A100-SXM4-80GB), the host that
  reproduces the released logs exactly.
* **Gate-signal fidelity, measured per model.** On Qwen2.5-1.5B the E8 runner
  reproduces released `D` at **0.00e+00** on 5/5 inputs with matching `T`. On
  Qwen2.5-14B it reproduces only to ~1e-4. Three controls localised that gap to
  pre-existing environment drift rather than anything E8 introduces: one-pass
  vs two-pass both land ~1e-4 (one-pass closer); sharded vs single-card are
  bit-identical to each other; and the attention patch itself is exactly
  neutral. At ~1e-4 against tau = 0.07 it cannot flip a gate call.
* **A reimplementation hazard, found the hard way.** Rewriting `build_prompt`
  rather than mirroring `gated_eviction.py` shifted `D` by 4e-4 through a
  single `\n\n` vs `\n` before the answer prefix. The runner now mirrors the
  canonical prompt exactly. Any new runner must be diffed against released `D`
  before its numbers are believed.
* **Qwen2.5-14B needs two cards.** At 4K it materialises 64.4 GB of attention
  tensors (48 layers x 40 heads) plus ~28 GB of weights, so it OOMs on one
  80 GB card in one-pass mode. E8 shards it with `device_map=auto`, which keeps
  it on the same one-pass path as the other three cells. Two-pass was rejected
  as the fix precisely because it changes what the scorer sees, which is the
  W9 degeneracy this round documents.
* Partial run files are **refused, not summarised**. `adakv_analysis.py`
  reports a cell as incomplete rather than averaging a short file.
* **Double-BOS tokenization.** The runner initially omitted
  `add_special_tokens=False` (which `gated_eviction.py:538` uses). Mistral and
  Llama chat templates already emit BOS, so a second was prepended: on Mistral
  that drifted `D` up to 1.13e-02 and **flipped 18 of 400 gate calls**. Qwen
  templates emit no BOS, so those cells were unaffected and hid the bug for the
  whole round. Fixed; post-fix Mistral reproduces the released log exactly.
  Any new architecture must be diffed against its released `T` and `D` first.
* **New model families must be added to `patch_families()`.** Without the
  patch the per-head mask silently does not apply and both arms become
  identical, which looks like a null result rather than an error.
