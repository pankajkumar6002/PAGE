# Pre-registration: B2, beyond RULER

Written before any B2 run. Date: 2026-11-07. Answers reviewer items S-Q2, G-W3 and G-Q1.

## Question

On long-context tasks other than RULER, does D select inputs better than a random gate at the same open rate, and does it predict per-input harm?

## What has already been looked at

Analysis A6 (internal analysis notes, not released) read the existing LongBench logs and found no selection over a random gate, with small N. These runs use different datasets, subtasks or models, so they are new data. Early-layer agreement a_early was noticed post hoc in A5, and these runs are its held-out test.

## Design

- **New datasets.**
  - QuALITY (`emozilla/quality`, validation split), multiple choice over long articles. 300 questions from a shuffle with seed 0, so they span many articles, on Qwen2.5-3B and Mistral-7B. This replaces "first 300 questions" from an earlier version of this file, changed before any run because the first 300 cover few articles.
  - LongBench-v2 (`zai-org/LongBench-v2`), multiple choice, contexts filtered to at most 32,000 tokens because longer ones do not fit, on Qwen2.5-3B and Mistral-7B. Usable N is reported.
- **LongBench.**
  - Missing subtasks 2wikimqa, musique and gov_report on Qwen2.5-14B, N up to 100.
  - lcc, repobench-p and hotpotqa on Qwen2.5-3B, N up to 100, with the same generic prompt as Table 12.
- **Eviction.** SnapKV, budgets {1, 1/2, 1/4, 1/8, 1/16}, sink 4, window 32, top-k 32. Only the plain arm is decoded. Gated outcomes follow from the gate identity, which held with zero violations on the released logs.
- **Scoring.** Multiple choice: the first standalone letter A to D in the prediction, with parse failures counted as wrong and reported. LongBench: substring match for parity, plus LongBench's official metric on stored predictions.

## Metrics

- **Primary.** Selection = PAGE minus a random gate at the same open rate, at b = 1/16 and 1/8, with a bootstrap over inputs, or over documents where questions share one.
- **Secondary.** Plain, full and PAGE accuracy; open rate; achieved cache; AUC(D) against harm; AUC(a_early) against harm; z-gate versions of the above.

## Predictions, recorded before running

- **P1.** Selection's interval includes zero on most (dataset, model) pairs. Basis: A6.
- **P2.** AUC(D) against harm stays below 0.65 on most pairs.
- **P3.** a_early does not clearly beat D against harm on most pairs. The perfect MK3 separation in A5 may be a NIAH prompt-format effect rather than an eviction signal. Low confidence.
- **P4.** PAGE minus plain is non-negative at b = 1/16 wherever the gate closes on some inputs and eviction harms some.

## Reporting

Every (dataset, model) pair is reported, including failures and any dataset that yields too few usable inputs. Nothing is dropped because it looks bad.

## Validation gate

On Qwen2.5-3B, the runner re-runs the first 5 inputs of LongBench qasper and must reproduce logged `drop` within 0.005, `T` exactly, and `pred_plain` on the first 200 characters at b = 1 and 1/4 on at least 4 of the 5 inputs.
