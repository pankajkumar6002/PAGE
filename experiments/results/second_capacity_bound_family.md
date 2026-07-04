# Second capacity-bound task family: validation attempt (LongBench) + fallback

Goal: reproduce, for a non-RULER task, the evidence the paper has for NIAH-MK3
(eviction monotonically destroys accuracy; per-input recovery rho ~ 0; head-agreement
drop D < tau = 0.07 so the gate classifies it capacity-bound a priori).

Candidate: **LongBench `passage_retrieval_en`** (identify which of ~30 paragraphs a
summary describes). Fallback checked: **LongBench `passage_count`**. Existing artifact
consulted for the within-RULER fallback: **`niah_multikey_2` @ 4K**
(`experiments/results/gated_4k_niah_mk_ablation.jsonl`, already reported in the paper's
distractor-sweep table).

## Protocol

- Model: Qwen/Qwen2.5-1.5B-Instruct, bf16, sdpa, GPU 3.
- Script: `experiments/scripts/longbench_gating.py` with `--two_pass` (memory-safe
  prefill; agreement/score computed on the 32-token observation window), obs_window=32,
  n_sink=4, top_k=32, tau=0.07. "Plain" arm = SnapKV-style eviction (obs-window
  attention scoring + sink retention), identical to the paper's RULER protocol.
- Budgets: {1.0, 0.5, 0.25, 0.125, 0.0625}.
- N: `passage_retrieval_en` N=100 (first 100 of 200; 0 skipped; T = 10,407-15,665
  tokens, no truncation needed under the script's 24K cap).
  `passage_count` N=47 (first 50 of 200 requested, 3 skipped for T > 24K;
  T = 5,587-23,849).
- Metric: repo convention (any gold answer substring of prediction) and a stricter
  regex metric (first "Paragraph (\d+)" / first integer in prediction vs gold).
  The two metrics agree exactly on every cell below.
- rho (Eq. (rho) in the paper): fraction of inputs where full-KV is wrong AND some
  budget b < 1.0 is correct. Operational dilution-prone threshold: rho >= 0.05.
- Raw outputs: `experiments/results/longbench_passret_qwen15b_n100.jsonl`,
  `experiments/results/longbench_passcount_qwen15b_n50.jsonl`;
  logs in `experiments/logs/longbench_pass{ret,count}_qwen15b_n*.log`.

## 1. passage_retrieval_en (N=100): accuracy vs budget

| budget | plain SnapKV acc |
|---:|---:|
| 1.0    | 0.290 |
| 0.5    | 0.280 |
| 0.25   | 0.290 |
| 0.125  | 0.280 |
| 0.0625 | 0.260 |

- **rho = 0.01** (1/100 inputs recovered under eviction).
- Per-input stability: of the 29 full-KV-correct inputs, 26-28 remain correct at every
  reduced budget (26/29 even at b = 0.0625, where only ~650-980 of ~10-15K tokens are
  kept). Accuracy is **flat, not monotonically destroyed** — the task is
  eviction-robust for this model, not capacity-bound.
- (An earlier N=30 run at budgets {1.0..0.125}, `longbench_passret_qwen15b.jsonl`,
  shows the same flat profile: 0.33/0.30/0.33/0.33, rho = 0.03.)

### Head-agreement drop stats (same inputs)

- mean D = **0.105** (std 0.010, min 0.079, max 0.131).
- Reference (Qwen2.5-1.5B @ RULER 4K, N=100/task, `drops_qwen15b_4k_n100.jsonl`):
  dilution-prone cluster D = 0.098-0.236 (fwe 0.111, multivalue 0.125, vt 0.161,
  qa_1 0.236, qa_2 0.213); capacity-bound MK3 D = 0.044.
- Gate at tau = 0.07: **0/100 inputs below threshold; gate-open fraction 1.00** —
  the gate classifies passage_retrieval_en as NOT capacity-bound (dilution-prone /
  evict-safe side). Its mean D sits squarely inside the dilution-prone cluster.
- Gate correctness: since accuracy is empirically robust to eviction down to
  b = 0.0625, the a-priori "safe to evict" classification is **correct** (gated
  predictions are identical to plain on all 500 cells).

**Verdict: passage_retrieval_en is NOT a second capacity-bound family.** Despite the
surface similarity to precise multi-needle retrieval (1-of-30 paragraph
identification), SnapKV's observation-window scoring concentrates on the queried
paragraph and eviction is nearly harmless. Consistent with the paper's claim that the
partition tracks near-tie distractor structure, not surface task category — the 30
paragraphs are not near-tie distractors under the summary query. rho = 0.01 < 0.05
also confirms it is not dilution-prone in the recovery sense (full-KV accuracy 0.29
leaves headroom, but no recovery occurs).

## 2. Fallback A: passage_count (N=47): floor case

| budget | plain SnapKV acc |
|---:|---:|
| 1.0    | 0.043 |
| 0.5    | 0.043 |
| 0.25   | 0.043 |
| 0.125  | 0.043 |
| 0.0625 | 0.043 |

- rho = 0.000. Drop stats: mean D = **0.054** (std 0.012, min 0.027, max 0.073);
  gate at tau = 0.07 classifies **43/47 (91.5%) of inputs capacity-bound**
  (gate-open fraction 0.085).
- Interpretation: the gate's a-priori prediction (capacity-bound: exact
  duplicate-counting over 30+ paragraphs plausibly needs the whole cache, and D sits
  below the MK3-side threshold) cannot be verified empirically on this model because
  Qwen2.5-1.5B is at floor: A_full = 0.043 (~chance), flat at every budget. With
  A_full ~ 0 the scaling formula gives rho ~ 0 trivially and there is no accuracy for
  eviction to destroy. **Unusable as an exemplar; not counter-evidence either.**
  (Suggestive but unconfirmed: it is the only non-RULER task measured so far whose
  mean D falls below tau.)

## 3. Fallback B: RULER niah_multikey_2 @ 4K (existing artifact, N=50)

From `gated_4k_niah_mk_ablation.jsonl` (same model, same protocol, same budgets):

| budget | plain SnapKV acc |
|---:|---:|
| 1.0    | 0.800 |
| 0.5    | 0.120 |
| 0.25   | 0.020 |
| 0.125  | 0.020 |
| 0.0625 | 0.000 |

- rho = 0.000; mean D = **0.122** (> tau, gate-open fraction 1.00).
- MK2 IS empirically capacity-bound (monotone collapse 0.80 -> 0.00, zero recovery)
  but the gate **misclassifies** it — exactly the single-near-tie boundary case the
  paper already documents (mean D = 0.122 co-located with FWE/multivalue).

## Verdict (citable)

**passage_retrieval_en is not a second capacity-bound family**: on Qwen2.5-1.5B
(N = 100, budgets 1.0 -> 0.0625), plain SnapKV accuracy is flat (0.29 -> 0.26) with
rho = 0.01, i.e., eviction neither destroys accuracy nor enables recovery. The
head-agreement gate classifies it correctly a priori: mean D = 0.105 (per-input min
0.079) lies inside the dilution-prone/evict-safe cluster (>= 0.098) and far from MK3
(0.044), so the gate opens on 100/100 inputs — and opening is the right call, since
eviction is empirically harmless. The fallback LongBench passage_count is a floor
case (A_full = 0.043 at every budget, N = 47): the gate does classify it
capacity-bound a priori (mean D = 0.054 < tau; 91.5% of inputs below threshold), but
the model has no accuracy to destroy, so the prediction is untestable on this model.
The only additional empirically capacity-bound family available therefore remains
within-RULER: niah_multikey_2 @ 4K (0.80 -> 0.00 monotone collapse, rho = 0.000,
N = 50), which the paper already reports as the single-near-tie boundary case that
the mean-D gate misclassifies (D = 0.122 > tau). Net: MK3 remains the paper's only
exemplar where both properties hold jointly (empirically capacity-bound AND gate-
classified capacity-bound); passage_count is a candidate for confirming the gate's
capacity-bound prediction on a non-RULER task, but doing so requires a model strong
enough to score above floor at full KV (e.g., a larger model), which is future work.

---
Runs executed 2026-07-02 on GPU 3;
commands:
`python experiments/scripts/longbench_gating.py --tasks passage_retrieval_en --out experiments/results/longbench_passret_qwen15b_n100.jsonl --model Qwen/Qwen2.5-1.5B-Instruct --budgets 1.0,0.5,0.25,0.125,0.0625 --max_examples 100 --max_new 32 --gpu 3 --two_pass`
(and `--tasks passage_count --max_examples 50` analogously).
