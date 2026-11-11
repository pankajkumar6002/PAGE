# Pre-registration: B1, scorer-based gates at a matched open rate

Written before any B1 run. Date: 2026-11-07. Answers reviewer items R-W4, R-Q1 and, through the comparison, R-W1.

## Question

Does the head-agreement drop D give a better gate than calibrated gates built from aggregated SnapKV, H2O or OBCache scores, or from activation change, at the same gate-open rate?

## What has already been looked at

On these same inputs, analyses A1, A2 and A5 (internal analysis notes, not released) already looked at D, a random gate, top-32 attention mass share, entropy drop and early-layer agreement. So B1 is confirmatory only for the new signals: H2O share, OBCache shares, SnapKV budget-specific share, and activation change. Results for early-layer agreement and entropy drop on these inputs are not held-out, and are labelled as such.

## Design

- **Models and inputs.** Qwen2.5-1.5B, Qwen2.5-3B, Mistral-7B at RULER 4K, tasks niah_multikey_3, vt, fwe, qa_1, N = 100 per task, same inputs as `gated_4k_*.jsonl`. Qwen2.5-14B at N = 50 per task if memory allows.
- **Signals.** Share of score mass retained at b in {1/16, 1/8, 1/4, 1/2} for SnapKV, H2O, OBCache value, OBCache key and OBCache joint. Activation change, KL divergence and top-1 agreement at the first decode step, at the same budgets.
- **Replay.** Outcomes come from the logged plain SnapKV arm. Joined on task and per-task index with context length asserted.
- **Primary metric.** Gain over plain SnapKV on the eight-budget grid when the signal closes the same number of inputs as D at tau = 0.07. Budget-specific signals use their b = 1/8 version for the primary comparison, chosen here before seeing results. Each signal's orientation is fixed by the direction that closes MK3, measured as AUC on the fitting model Qwen2.5-1.5B and applied to the others. Hindsight orientation per model is reported as a secondary, labelled favourable to the signal.
- **Comparison.** D minus signal, with a task-stratified input bootstrap (2,000 replicates). Random gate at the same open rate from A2.
- **Secondary.** AUC for MK3 with and without niah_multivalue where available; AUC against harm, pooled and within-task, at b = 1/16 and 1/4.

## Decision rule

- D "has an advantage over signal X" if D minus X has a 95% interval above zero on at least two of the three main models.
- X "has an advantage over D" if the interval is below zero on at least two of the three.
- Otherwise: "no clear advantage either way".

## Predictions, recorded before running

- **P1.** D has an advantage over the SnapKV budget-specific share. Basis: A5's top-32 share result.
- **P2.** D has an advantage over the H2O share.
- **P3.** OBCache shares do no better than the SnapKV share, because they score tokens and were not designed as input-level signals. No strong prior.
- **P4.** Activation change at b = 1/8 is competitive with D. It measures the effect of eviction directly, at the cost of one extra forward pass per budget. Low confidence.
- **P5.** Within-task AUC against harm includes 0.5 for most signals, as it does for D.

## Reporting

All signals and all models are reported, including any prediction that fails. A runner must first reproduce logged `drop` and `T` on the validation sample. If the validation fails, the run is discarded and the failure is reported.

## Validation gate

For the first 5 inputs of each task on each model: `T` equals the logged value, and |D - logged drop| is at most 0.005. The paper reports a host effect of 0.0035 on mean D, so this tolerance allows for it.
