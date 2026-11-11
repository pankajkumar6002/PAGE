# Pre-registration: B7, PAGE on L-Eval

Written before any B7 run, including the dry run. Date: 2026-11-09. Answers reviewer items S-Q2, G-W3 and G-Q1 (beyond RULER), at the user's request to add L-Eval (An et al., ACL 2024).

## Question

On L-Eval's closed-ended tasks, does D select inputs better than a random gate at the same open rate, and does it predict per-input harm?

## What has already been looked at

- B2, B2b and B2c (QuALITY, LongBench-v2, six LongBench subtasks) found no selection beyond a random gate outside RULER, and D near chance against per-input harm (internal analysis notes, not released, B2 and B2c).
- L-Eval's `quality` task is drawn from QuALITY, which B2 used. The overlap of articles with B2's 300-question sample will be reported. Everything else in L-Eval is new to this project.
- For setting up this run I read L-Eval's data format (task sizes, answer formats, context lengths in Qwen2.5 tokens), its prompt code (`Baselines/LEval_config.py`, `Baselines/llama2-chat-test.py`) and its scoring code (`Evaluation/auto_eval.py`, `Evaluation/em.py`), at GitHub commit `cd34b050`. No model has been run on L-Eval.

## Data

- L-Eval jsonl files from the Hub repo `L4NLP/LEval`, commit `43b9dbf06c239360b183c76cb5d90ebe4f5e3070`, downloaded as data to `data/leval/` and hashed in `data/leval/SHA256SUMS`. The dataset's loading script is not run.
- **Tasks:** the closed-ended ("Exam") tasks tpo, quality, coursera, topic_retrieval_longchat, gsm100 and sci_fi. Each instruction of a document is one question.
- **Length cap:** questions whose full prompt exceeds 16,000 tokens are excluded, not truncated. This keeps the paper's 4K to 16K range and fits one A100. With all three tokenizers this leaves 920 questions in the L-Eval arm (tpo 269, quality 202, coursera 163 of 172, topic_retrieval 150, gsm100 100, sci_fi 36 of 64) and 634 in the quote arm. codeU is excluded because none of its inputs fits (20K to 31K tokens).
- **Open-ended tasks are not run.** L-Eval's authors recommend its exam-style metrics over n-gram metrics, and LongBench QA with F1 is already in B2.

## Design

- **Models:** Qwen2.5-3B-Instruct, Mistral-7B-Instruct-v0.3, Qwen2.5-14B-Instruct, as in B2 and B2c.
- **Arms.**
  - **L-Eval arm (primary):** L-Eval's own system prompts (`get_sys_prompt`, verbatim) and message formats (`llama2-chat-test.py`), through each model's chat template. For tpo, quality, coursera and sci_fi the user turn is "Document is as follows. {document} \nQuestion: {inst}.  Please directly give the answer without any additional output or explanation ", and "Answer:" follows the generation prompt, where L-Eval puts "\nAnswer:" after `[/INST]`. For topic_retrieval and gsm100 it is "{document}\n\n{inst}". Documents are concatenated, not passed through `str.format`. Maximum new tokens: 32 for the option tasks and sci_fi, 64 for topic_retrieval, 256 for gsm100 (L-Eval uses 512 everywhere; these caps cover its answer formats).
  - **Quote arm (secondary):** tpo, quality and coursera only, with B2b's quote-then-answer instruction (coursera allows several letters) and 128 new tokens. B2 showed that answer-only multiple choice hides eviction harm. This arm checks the same here.
- **Eviction:** SnapKV, budgets {1, 1/2, 1/4, 1/8, 1/16}, sink 4, window 32, top-k 32, SDPA with the two-pass scoring prefill, bf16, greedy, exactly as in B2 (`longbench_gating.py`, unchanged). Only the plain arm is decoded. Gated outcomes follow from the gate identity.
- **Scoring:** L-Eval's exam scoring, reimplemented from `auto_eval.py` and `em.py`: option letters by `process_output_mc` and `process_gt_mc`, then `exact_match_score` (1, 0.25 for a strict subset of coursera's gold letters, or 0); gsm100 by `process_math` then numeric match; topic_retrieval by case- and space-insensitive substring; sci_fi by the first true or false word, scored on the document-based (loyalty) half only, since L-Eval's no-document fact half cannot be affected by eviction. A question is correct when its L-Eval score is 1. The quote arm reads the last "Answer: X" (B2b's pattern for single answers; letter groups for coursera). Parse failures count as wrong and are reported.
- **Runner:** `experiments/scripts/b7_leval.py`. It writes a question only when all five budgets are done, and on a restart it keeps complete questions and redoes the rest.
- **Hardware:** the second server (one A100 80GB PCIe), with the same Python environment copied to the same path and the models copied from the main server's cache.

## Validation gate, before any L-Eval run

All on the second server, through the B2 runner `b2_gating.py`, unchanged, with SDPA and two-pass, as B2 ran:
1. **V1 (B2's gate):** Qwen2.5-3B, LongBench qasper, first 5 inputs at b = 1 and 1/4, against the released log `longbench_qwen3b.jsonl`.
2. **V2 to V4 (cross-machine):** first 5 QuALITY questions at all five budgets against B2's own logs from the main server: `b2_quality_qwen3b`, `b2_quality_mistral7b`, `b2c_quality_qwen14b_letter`.

Each must reproduce `T` exactly, `drop` within 0.005, and `pred_plain` on the first 200 characters at b = 1 and 1/4. **Any mismatch stops the queue** and is reported. Nothing is run on L-Eval until the gate passes.

## Dry run

After hashing and after the validation gate: Qwen2.5-3B, two questions per task in each arm, and the longest questions on Qwen2.5-14B for peak memory. Outputs go to `experiments/results/dryrun/` and are never analyzed. Any change to the runner or the analysis after the dry run is recorded below as an amendment and re-hashed before the main runs.

## Metrics

- **Primary:** selection = PAGE minus a random gate at the same open rate, pooled over the six tasks, at b = 1/16 and 1/8, with the z-gate. Its threshold is mean D + (-0.69) SD over all of a run's questions, pooled across tasks, so it needs no task labels. The bootstrap resamples documents within each task, 2,000 replicates. For gsm100, which has one shared prompt, each question is its own unit.
- **Secondary:**
  - fixed tau = 0.07;
  - each task separately, with the pooled threshold, and with the threshold standardized within the task, as B2 did per dataset;
  - plain, full and PAGE accuracy, open rate, achieved cache;
  - AUC(D) against harm, pooled and within task;
  - L-Eval-style mean scores.

## Predictions, recorded before running

- **P1.** In the L-Eval arm, pooled z-gate selection's interval includes zero on most (model, budget) units at 1/16 and 1/8. Basis: B2.
- **P2.** In the L-Eval arm, AUC(D) against harm at 1/16 is below 0.65 on most (task, model) units with at least 10 harmed and 10 surviving questions. Basis: A1, B2.
- **P3.** Fixed tau opens on under 20% of questions on most (model, arm) runs. Basis: B2, where it opened on 0% to 12%.
- **P4 (low confidence).** topic_retrieval_longchat does not have the lowest mean D of the six tasks on most models. Basis: the paper's account, where the capacity-bound class needs near-tie distractors. Topic names in L-Eval's conversations are distinct.
- **Q1.** In the quote arm, at least 15% of full-correct answers are lost at 1/16 on most (task, model) units. Basis: B2b's 19% to 33%.
- **Q2.** In the quote arm, pooled z-gate selection's interval includes zero on most (model, budget) units. Basis: B2b.

## Reporting

- Every (model, arm, task) is reported, including tasks with few usable questions or near-zero full-cache accuracy. Nothing is dropped because it looks bad.
- Analysis: `experiments/scripts/analyze_b7.py`, hashed with this file. It reads a run only when its log ends with the runner's completion line and the row count matches.

## Amendments

### Amendment 1 (2026-11-09, before any main run)

**What happened under the original hash.** The first check ran on the second server from 11:55 to 12:01 UTC on 2026-11-09.
- **Validation:** passed, 4 of 4. Every check reproduced `T`, D exactly (|dD| = 0) and the prediction at every budget.
- **Dry run:** finished, 10 of 10 jobs. I read its outputs to check prompts, parsing and memory: 2 questions per task and arm on Qwen2.5-3B, and the 3 longest coursera questions that fit on Qwen2.5-14B, which peaked at 37.3 GiB. These questions are also in the main runs, since they are the first of each task. The dry-run files are kept in `experiments/results/dryrun/` and never analyzed.
- **Review:** an independent review agent read the code against L-Eval's. It fuzzed the scoring with 300,000 strings against its own transcription and found no mismatch, and it found the issues below.

**Changes.**
1. **gsm100 output cap: 256 tokens to 512, L-Eval's own value.** In the dry run, 3 of 10 gsm100 outputs on Qwen2.5-3B hit 256 tokens before the final answer. That would count truncation as eviction harm.
2. **Parse failures in the L-Eval arm's option tasks are now flagged.** An output that is empty or has no A to D letter is flagged; the score is unchanged. This corrects "Parse failures count as wrong" in the Scoring paragraph:
   - In the L-Eval arm, every task is scored exactly as L-Eval scores it. That includes its fallback answer "A" for an option output with no letter, which is correct when the gold is A. Such outputs are reported as parse failures.
   - In the quote arm, a missing "Answer:" line counts as wrong.
3. **Quote arm, single-answer tasks:** B2b's pattern now has a guard, so "Answer: Based on ..." is no longer read as B. This is a deliberate departure from B2b. Coursera's parser already rejected such words.
4. **Runner robustness:**
   - on a restart it skips a cut-off last line and rewrites the log atomically;
   - out-of-memory errors raised by cuBLAS are handled like other out-of-memory errors.
5. **Analysis:**
   - one random generator per run, so a run's intervals do not depend on which other runs are complete;
   - a run table (B7-T0) with the counts over the cap and out of memory;
   - per-task parse-failure columns in B7-T2;
   - the quality overlap with B2, computed in the script.
6. **Queue:**
   - each validation check runs once in a fresh log, and must give exactly the expected number of checks with no skip or out-of-memory;
   - main runs need a gate file that a check deletes at its start and writes only after validation, dry run and the hash check all pass;
   - `pkill` is limited to my own user;
   - a job gets up to 6 attempts, and giving up is logged;
   - the dry-run archive path is fixed.
7. **Corrections to the text above:**
   - **The 16,000-token cap** is an upper bound only. Prompts run from about 3,000 tokens (tpo) to 16,000. It does not "keep the paper's 4K to 16K range".
   - **Completeness:** the analysis reads the last completion line in a run's log. `launch.sh` appends its own line after it, so the log does not end with it.
   - **V1 is stricter than B2's gate.** B7 requires all 5 inputs to match; B2 allowed 4 of 5.
   - **Quality overlap with B2:** all 15 of L-Eval's quality articles are among the 110 articles of B2's QuALITY sample, and 24 of its 202 questions are among B2's 300. This was matched on a 200-character passage from the middle of each article, after a first-400-character match gave 4 because of header differences. L-Eval's quality task therefore adds L-Eval's prompt and 178 new questions, not new documents.

**Rerun.** The check (validation and dry run) is rerun under the amended hash before the main runs, and the main runs start only from its gate file. No prediction, metric, model, task or budget changed.

### Amendment 2 (2026-11-09, during the first main run; execution only)

At the user's request, the main runs move from the serial queue (`b7_queue.sh main`) to a watchdog that runs up to two of our jobs at once on the second server's card: a watchdog script (not released), a copy of the main server's reviewed `watchdog2.py`, launching through `launch_multi.sh`, with the queue in a queue file (not released).
- **Same rules as before:**
  - each launch re-checks this hash;
  - a job starts only when the card has its need plus 10 GB free;
  - main runs need the gate file from the last check;
  - other users' jobs are never touched.
- **New rules:**
  - at most two of our jobs per card;
  - a job that is still loading keeps its whole need reserved;
  - when the card stays under 2 GB free, only our most recently started job yields;
  - B7 output is never set aside, so a stopped or retried run resumes from its last complete question.
- **Tested before use** in a simulation with a stub launcher and a fake `nvidia-smi`: two jobs at a time in queue order, a low-memory yield of the newest job with its output kept, reservations under tight memory, and no start without the gate file.
- **What happened to the first main run (disclosed).** `b7_leval_qwen3b` had started under the serial queue and was near its end when the switch was made.
  - I stopped only the serial queue, meaning to let that run finish. But I left the run in the watchdog's queue, so at 13:25:49 UTC the watchdog started a second copy with the same output file.
  - That copy's resume step rewrote the file atomically, keeping the 869 complete questions. The first run's later rows would therefore have gone to the replaced file.
  - I stopped the second copy about 25 s after it started, and then the first run, about a minute later. Only that minute of work was lost.
  - The output file was then checked: 4,345 rows, 869 complete questions, no duplicate, partial or unreadable rows.
  - Both copies ran the same runner under the same hash. The run resumes from those 869 questions under the watchdog.
- **Guards added to the watchdog after this:**
  - it never starts a job while another process of ours has `--out` with that job's output file among its arguments;
  - a GPU process of ours that it did not start counts as one of our two jobs on the card.
- **No change to models, data, prompts, scoring, budgets, metrics or predictions.**
- **Sharing a card should not change any output.** The main server's B2 runs shared their cards with other users' jobs, and the second server reproduced them exactly. To confirm it here, check V2 (QuALITY, Qwen2.5-3B, against B2's log) is re-run while a main job shares the card (`b7_val_concurrent_quality_qwen3b`), and must again reproduce `T`, D and every prediction.

### Amendment 3 (2026-11-09, before any B7 result was looked at): two more models

At the user's request ("run other models too if needed"), two models are added. All five B7 models were Qwen2.5 or Mistral, which is the coverage reviewer GqTU criticizes (G-W4, G-Q2).
- **Qwen3-14B,** with thinking off in the chat template as in B3. B3 showed PAGE reproduces on RULER on this model, so this tests whether that carries beyond RULER.
- **Llama-3.1-8B-Instruct,** a third model family, which the paper's appendix covers on RULER.

**What has been looked at.** No B7 accuracy, harm, selection or AUC from any main run, at the time of this decision. My status checks printed the runner's progress lines, which show single questions' D and T only.

**Design.** Everything is as pre-registered and amended: tasks, 16,000-token cap, both arms, budgets, scoring, metrics.
- Both models fit 920 questions in the L-Eval arm and 634 in the quote arm, the same counts as the original three.
- **Runner changes, both off by default:**
  - `--no_thinking` passes `enable_thinking=False` to the chat template. It is used for Qwen3 only; Qwen2.5's prompt is byte-identical with or without it.
  - `--per_task_limit` takes the first N questions of each task, for dry runs.
  - The queued runs of the original models therefore behave exactly as before.
- **Dry run first:** 2 questions per task and arm on each new model, on the second server. I inspect it, as before, and its outputs are not analyzed. The main runs of these models start only after that, through a marker file I create.
- **The analysis reports these two models separately.** The same predictions P1 to P4, Q1 and Q2 are counted over their units in B7-T7. B7-T6's verdicts stay on the three pre-registered models, with unchanged random seeds.
