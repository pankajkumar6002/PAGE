# Pre-registration: B2c, QuALITY on Qwen2.5-14B, both prompt arms

Written on 2026-11-07 at 22:40, after B2 and B2b finished and before any B2c run. It answers one question left open by B2b: does the only positive selection result beyond RULER replicate on a second, larger model?

## Why this exists

Across B2 and B2b, selection beyond a random gate is not shown on real documents, with one exception: Mistral-7B, QuALITY, quote-then-answer prompt, z-gate, b = 1/8, selection +0.026 [+0.005, +0.047], bootstrapped by document (`analysis-results.md`, B2-T1). One positive interval among many is what chance alone can produce, so it needs a replication before it can be cited as support.

## Design

Identical to B2 (letter arm) and B2b (quote arm) except for the model. Same runner (`experiments/scripts/b2_gating.py`, unchanged), same data (`emozilla/quality` validation, seed-0 shuffle, first 300 questions), same budgets {1, 1/2, 1/4, 1/8, 1/16}, same SnapKV eviction through `longbench_gating.py` with `--attn_impl sdpa --two_pass`, same scoring and parse rules.

- **Model.** Qwen/Qwen2.5-14B-Instruct, the largest model with B2 LongBench logs.
- **Quote arm (primary).** `--mcq_style reasoning`: quote the supporting sentence, then `Answer: X`, max 128 new tokens.
- **Letter arm (secondary).** Letter only, max 8 new tokens. It is run so that harm can be compared between arms on the same model, as in B2-T3.

## Metrics

As B2 and B2b: accuracy and selection against a random gate at the same open rate at b = 1/8 and 1/16, bootstrapped by document; AUC(D) and AUC(a_early) against harm; harm at 1/16 by arm; open fractions; for fixed tau and the z-gate (theta_z = -0.69, standardized over the cell's inputs).

## Predictions, recorded before running

- **P1, the replication test.** The Mistral result replicates only if the quote arm's z-gate selection at b = 1/8 is positive with a 95% interval that excludes zero. We predict that it does not replicate.
- **P2.** Harm at b = 1/16 is higher in the quote arm than in the letter arm, as on both B2b models.
- **P3.** In the quote arm, selection's interval includes zero at both b = 1/8 and 1/16 under both gates, apart from the P1 cell, and AUC(D) against harm is below 0.65.
- **P4.** Fixed tau opens the gate on at most 12% of inputs in both arms, as on QuALITY and LongBench-v2 for the B2 models.

## Reporting

Both arms are reported whatever they show. If the quote arm's z-gate selection at b = 1/8 is positive with an interval excluding zero, the response reports that the Mistral result replicated on Qwen2.5-14B. Otherwise it says the Mistral result did not replicate, and does not cite it as support.
