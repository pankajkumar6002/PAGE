# Pre-registration: B8, PAGE on realistic near-tie retrieval

Written before any B8 run, including the validation and dry run. Date: 2026-11-10. Requested by the user after the ARS panels (internal notes, not released). All ten simulated reviewers named one result that could change their rating: PAGE beating a random gate at the same open rate on a realistic, independently chosen capacity-bound task. Reviewer items: S-Q2, G-W3, G-Q1, R-Q1.

## Question

When a realistic near-tie retrieval task is part of a realistic mixed workload, does PAGE's gate close on it without task labels? Does that give positive selection over a random gate at the same open rate?

## Why these datasets (chosen before any run, from the paper's own account)

The paper's account (Appendix A.2.4) says eviction harm is capacity-bound when the answer must be told apart from near-tie distractors. NIAH-MK3 does this with templated keys. Two public tasks have the same structure without RULER's templates. Both were chosen for that structure alone, before any of their outcomes was seen.

- **nq30:** NQ-open multi-document QA with 30 retrieved Wikipedia passages (Liu et al., "Lost in the Middle", TACL 2024).
  - Each question has one gold passage and 29 passages retrieved for the same question that lack the answer. The distractors are real text and topically near-identical.
  - This is the most realistic of the two: a retrieval-augmented generation (RAG) prompt.
- **babilong:** BABILong qa2 and qa3 at 4k and 8k (Kuratov et al., NeurIPS 2024 Datasets and Benchmarks).
  - bAbI facts are placed in PG19 book text. A question needs two or three facts about the same people, items and rooms, among distractor facts with the same form.
  - The haystack is natural text, but the facts are templated. It is reported as semi-synthetic.

**Considered and not used.**
- RepoQA: its release file returns 404.
- HELMET RAG: the same structure as nq30, with a multi-GB archive.
- InfiniteBench kv_retrieval: synthetic UUIDs, and over 100K tokens.
- LongBench passage_retrieval_en: B2 found it evict-robust.

## What has already been looked at

- **Earlier results.** B2, B2b, B2c (QuALITY, LongBench-v2, six LongBench subtasks) and B7 (six L-Eval tasks): no selection beyond a random gate outside RULER, and D near chance against per-input harm. In B7's L-Eval arm the pooled z-gate's selection at 1/16 was -0.008, -0.003 and -0.015 on the three models (`b7_analysis.md`, B7-T1). I have read these results.
- **LongBench hotpotqa,** a multi-document QA task with distractors, was null in B2. That is evidence against this test's hypothesis.
- **No model has been run on nq30 or babilong.**
  - Done for setup only: their prompts were tokenized to measure lengths (2,534 to 9,552 tokens; none over the cap), and their prompt and scoring code was read.
  - The analysis script was smoke-tested on fake inputs made by relabelling B7 L-Eval rows. Those outputs carry no information about B8.

## Data

- Raw files in `data/b8_raw/`, with sources in `SOURCES.md` and hashes in `SHA256SUMS`. No downloaded code is executed.
- **nq30:** 350 of the 2,655 questions, chosen as the first 350 of a seed-0 permutation (`numpy.random.RandomState(0)`). The gold position cycles through 0, 4, 9, 14, 19, 24, 29 in sample order, so each position gets 50 questions.
  - Prompt: Liu et al.'s `qa.prompt`, verbatim, with passages formatted as in their `prompting.py`.
  - Score: their `best_subspan_em`, verbatim. A question is correct when it scores 1.
- **babilong:** all 400 questions of qa2 and qa3 at 4k and 8k.
  - Prompt: BABILong's `DEFAULT_PROMPTS` and `DEFAULT_TEMPLATE`, verbatim, checked equal to the source.
  - Score: their `compare_answers` with their qa2 and qa3 labels.
- Both prompts are passed as the user turn of each model's chat template, with no system prompt. The cap is 16,000 tokens, which no question exceeds.

## Design

- **Models:** Qwen2.5-3B-Instruct, Mistral-7B-Instruct-v0.3 and Qwen2.5-14B-Instruct. These are the three models with a complete B7 L-Eval-arm run, which serves as the background workload.
- **Runner:** `experiments/scripts/b8_neartie.py`. It calls `longbench_gating.py` unchanged for prefill, D, SnapKV masks and decoding:
  - SDPA with the two-pass scoring prefill, bf16, greedy decoding;
  - sink 4, window 32, top-k 32;
  - budgets {1, 1/2, 1/4, 1/8, 1/16}.

  These are exactly the settings of B2 and B7. Only the plain arm is decoded; gated outcomes follow from the gate identity. Maximum new tokens: 64 for nq30, 32 for babilong.
- **Background workload:** the model's complete B7 L-Eval-arm log, all 920 questions over six tasks, under its own scoring (correct means an L-Eval score of 1). It is used unchanged and has been analysed already (see above).
- **Hardware:** the main server's four A100 80GB cards, alongside other users' jobs.
  - At most one of our jobs per card, including jobs from the user's other projects.
  - A job starts only when the card has its need plus 10 GB free.
  - Other users' jobs are never touched.
  - The second server's card is busy with B7. B8 jobs go there only if a main-server card stays unavailable, and that would be recorded as an amendment.

## Validation gate, before any B8 run

1. **V1:** `b2_gating.py`, unchanged, reproduces B2's QuALITY log `b2_quality_qwen3b.jsonl` on its first 5 questions at all five budgets (SDPA, two-pass), with `T` exact, D within 0.005 and `pred_plain` equal on the first 200 characters at b = 1 and 1/4. All 5 must match.
2. **V2:** the B8 runner, run twice on the same 2 questions per task on Qwen2.5-3B, must give identical D, `T` and predictions.

**Any failure stops the queue** and is reported.

## Dry run

After hashing and validation: Qwen2.5-3B, 3 questions per task, plus the 3 longest babilong questions on Qwen2.5-14B for peak memory. I read the outputs only to check prompts, parsing and memory. They go to `experiments/results/dryrun_b8/` and are never analyzed. Any change after the dry run is recorded below as an amendment and re-hashed before the main runs.

## Metrics

**Primary.** For each near-tie dataset X (nq30, babilong) and each model:
- The workload is X plus the model's 920 L-Eval questions. Its near-tie share is the natural one from the sample sizes: 350/1270 = 0.28 for nq30, 400/1320 = 0.30 for babilong.
- Selection is PAGE minus a random gate at the same open rate, with the z-gate (mean D + (-0.69) SD over the whole workload, no task labels) at b = 1/16.
- The bootstrap uses 2,000 replicates. Within each task it resamples documents, where each nq30 or babilong question and each gsm100 question is its own document. The z threshold is re-fit in every replicate. The two-sided bootstrap p is 2 x min(share of replicates at or below 0, share at or above 0).
- Holm correction runs across the 6 primary tests (2 datasets x 3 models).

**Pre-registered win.** PAGE beats the random gate on dataset X if selection is positive with Holm-adjusted p < 0.05 on at least 2 of the 3 models. Anything else is reported as no win.

**Secondary,** all reported whatever they show:
- b = 1/8, and the fixed tau = 0.07 gate, which does not depend on the workload mix.
- **Gate-free capacity-bound screen:** X is capacity-bound for a model if plain SnapKV at 1/16 loses at least 15% of the answers correct at full cache. This needs no D. The primary test is reported whether or not the screen passes, and the screen's result is shown next to it.
- Accuracy at every budget, and the harm count.
- X's mean D, ranked against the six L-Eval tasks.
- The closed fraction on X and on L-Eval.
- AUC(D) against harm within X.
- **Share sweep:** X subsampled to 10%, 25% and 50% of the workload, mean selection over 200 draws. This answers the objection that theta_z encodes MK3's 25% share in the paper's RULER pool.
- **X alone,** with the z threshold within X: per-input selection.

The analysis is `experiments/scripts/analyze_b8.py`, hashed with this file. Its seeds are fixed in the script.

## Predictions, recorded before running

- **P1 (moderate confidence).** nq30 passes the harm screen on at least 2 of 3 models. Basis: multi-document QA degrades under heavy eviction in published KV-compression studies, and 1/16 keeps about 300 of about 5,000 tokens.
- **P2 (moderate confidence).** babilong passes the harm screen on at least 2 of 3 models. Basis: qa2 and qa3 need two or three facts spread through the haystack.
- **P3 (low confidence, about 30%).** nq30's mean D is the lowest of the seven tasks in its workload on at least 2 of 3 models. Basis: the paper's near-tie account supports it. B2's hotpotqa, with no low D and no selection, argues against it.
- **P4 (low confidence, about 30%).** The primary test is a win on nq30. I expect no win more likely than a win, on the evidence of B2, B7 and A1.
- **P5.** X alone, z within X: the selection interval at 1/16 includes zero on most (dataset, model) units. Basis: A1's within-task AUC, mean 0.46.

## Reporting

- **Everything is reported.** Every (dataset, model) unit, the predictions that fail, and a no-win outcome. A win on one dataset is never reported without the other dataset's result and the share sweep.
- **Status if it wins.** A win would be new, pre-registered evidence of selection beyond RULER. It would not show per-input prediction, which P5 and A1 address.
- **Status if it does not win.** It would be one more pre-registered null, reported alongside B2 and B7.

## Amendments

### Amendment 1 (2026-11-10, after the check and before any main run; execution only)

**Check.**
- **V1 passed:** 25 of 25 rows reproduce B2's QuALITY log, with `T` exact, |dD| = 0 and every prediction equal.
- **V2 passed:** two runs of the B8 runner gave identical rows, 15 on nq30 and 30 on babilong.
- **Dry run:** Qwen2.5-3B, 3 questions per task, and Qwen2.5-14B on babilong. I read its outputs only to check prompts, parsing and scoring, and changed nothing. Its rows stay in `experiments/results/dryrun_b8/` and are never analyzed.

**Corrections to the text above.**
- **The Qwen2.5-14B dry run was not a peak-memory test.** `--limit` takes the first questions per task, which are 4k BABILong questions. Peak memory is covered by B7, where Qwen2.5-14B on prompts up to 16,000 tokens peaked at 37.3 GiB; B8's longest prompt is 9,552 tokens.
- **Memory need.** The queue's need for Qwen2.5-14B was 42 GB. With the 10 GB margin that exceeds the 48.5 GB free beside other users' jobs, so the dry run waited 3.5 hours for a card. The need is now 38 GB, from B7's measured peak, and the 10 GB margin is unchanged.

**No change to data, prompts, scoring, models, budgets, metrics or predictions.**
