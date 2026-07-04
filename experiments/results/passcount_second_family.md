# LongBench passage_count on a larger model: capacity-bound retest

Follow-up to `second_capacity_bound_family.md` §2 (fallback A), which found LongBench
`passage_count` GATE-PREDICTED capacity-bound (mean D = 0.054 < tau = 0.07 on
Qwen2.5-1.5B, gate closing on 91.5% of inputs) but UNTESTABLE there because full-KV
accuracy was at floor (A_full = 0.043 ~ chance): with no accuracy, eviction had nothing
to destroy. Here we rerun on a model whose full-KV accuracy clears floor.

## Protocol

- Model: **Qwen/Qwen2.5-14B-Instruct**, bf16, sdpa, GPU 3 (~38 GB resident).
- Script: `experiments/scripts/longbench_gating.py --two_pass` (memory-safe prefill;
  agreement/score on the 32-token observation window), obs_window=32, n_sink=4,
  top_k=32, tau=0.07. "Plain" arm = SnapKV-style eviction (obs-window attention scoring
  + sink retention) — identical protocol to the paper's RULER runs and the 1.5B run.
- Budgets: {1.0, 0.5, 0.25, 0.125, 0.0625}. `--max_new 32`.
- **N = 95** usable (first 100 of the 200-example test split requested; 5 skipped for
  T > 24 K). T range 5,079-23,849 tokens. Run time 731 s (~12 min).
- Metrics: repo substring convention AND a stricter exact-integer metric (first integer
  in prediction == gold count). Exact-int is the correct metric for a counting task
  (substring spuriously matches, e.g. "1" in "13"); reported as primary. Both give
  identical A_full.
- Raw: `experiments/results/longbench_passcount_qwen14b.jsonl`;
  log: `experiments/logs/longbench_passcount_qwen14b_n100.log`.

## 1. Full-KV accuracy: now above floor

- **A_full = 0.158** (15/95, exact-int; identical under substring).
- Reference: Qwen2.5-1.5B floor A_full = 0.043 (~chance); majority-guess baseline on
  this N (always "5") = 0.084. So 14B clears floor (~4x the 1.5B value, ~2x majority,
  above the ~0.15 testability bar). The signal is real but weak.

## 2. Accuracy vs budget (plain SnapKV, exact-int)

| budget | plain acc | gated acc | full-KV-correct retained |
|---:|---:|---:|---:|
| 1.0    | 0.158 | 0.158 | 15/15 |
| 0.5    | 0.147 | 0.158 | 14/15 |
| 0.25   | 0.147 | 0.158 | 14/15 |
| 0.125  | 0.137 | 0.158 | 13/15 |
| 0.0625 | 0.126 | 0.158 | 12/15 |

- Accuracy is **NOT destroyed**: 0.158 -> 0.126 across a 16x cache reduction (only ~6%
  of tokens kept at b = 0.0625). Of the 15 full-KV-correct inputs, **12/15 survive at
  b = 0.0625** — a gentle erosion of 3 examples (within binomial noise), the opposite of
  the collapse-to-zero of the paper's capacity-bound exemplars (MK3 ~0.99 -> 0.00,
  MK2 0.80 -> 0.00). A task that genuinely needed the whole cache to count 30 passages
  could not keep 80% of its correct answers after discarding 94% of tokens.
- **rho = 0.000** (exact-int; 0 inputs recovered at any b < 1.0). (Substring metric
  reports rho = 0.011 from one spurious digit match; discard.)
- gated column is flat at 0.158 because the gate is closed on every input (§3), so the
  gated policy never evicts and equals full-KV everywhere.

## 3. Head-agreement drop / gate

- **mean D = -0.0128** (std 0.0117, min -0.0474, max 0.0145) — slightly negative (late
  layers agree marginally more than early), even further below tau than on 1.5B (0.054).
- **95/95 inputs below tau = 0.07; gate-open fraction = 0.000.** The gate predicts
  **capacity-bound (do-not-evict) on 100% of inputs.**

## Verdict — case (c): eviction-robust; gate's capacity-bound prediction is NOT confirmed

On the larger model where full-KV accuracy clears floor (A_full = 0.158 vs 0.043 on
1.5B), **eviction does not destroy accuracy** — plain SnapKV accuracy is statistically
flat under 16x eviction (0.158 -> 0.126; 12/15 full-KV-correct inputs retained at
b = 0.0625) with rho = 0.000. This is task-background case **(c)**: above floor, but
eviction does not destroy accuracy, so passage_count is **eviction-robust, NOT
capacity-bound** (robust by preservation, hence rho = 0 rather than the >= 0.05 of the
dilution-prone/recovery variant).

The defining empirical signature of capacity-boundedness — accuracy collapse under
eviction — is **absent**. Note that "rho ~ 0 and gate closed" does NOT by itself imply
capacity-bound: capacity-bound tasks (accuracy destroyed, no recovery) and evict-robust
tasks (accuracy preserved, no recovery) BOTH have rho ~ 0. The discriminator is whether
accuracy is destroyed, and here it is not. So passage_count is a **false positive of the
head-agreement gate**: the gate closes on all 95 inputs (mean D = -0.013 < tau),
predicting do-not-evict, yet 16x eviction is empirically safe. Where the 1.5B floor left
this prediction untestable, the 14B test refutes it.

Consequence for the paper: the passage_count limitation should be upgraded to a
**negative result**, not to a confirmed second capacity-bound family. **MK3 remains the
paper's sole exemplar where empirical capacity-boundedness and the gate's capacity-bound
classification coincide.** (This is the conservative dual of the MK2 boundary case the
paper already documents: there the gate OPENS on a task that is empirically
capacity-bound; here the gate CLOSES on a task that is empirically evict-safe.)

Caution on power: the signal is weak (A_full = 0.158, 15 correct), so the retest is
underpowered to see a small accuracy collapse. But it is well powered against a *large*
one: an MK3-style collapse would drive the 15 correct answers toward the 0.084 majority
baseline, and instead 12/15 survive at 16x — the strong-collapse hypothesis is excluded.

### Exact sentence the paper should use (replacing the passage_count "future work" note)

> Retesting LongBench passage_count on Qwen2.5-14B-Instruct, where full-KV accuracy
> clears floor (A_full = 0.158 vs 0.043 on 1.5B; N = 95, budgets 1.0 -> 0.0625), shows it
> is eviction-robust rather than capacity-bound: plain SnapKV accuracy is statistically
> flat under 16x eviction (0.158 -> 0.126, with 12 of 15 full-KV-correct inputs retained
> at b = 0.0625 and rho = 0.000), so the head-agreement gate's a-priori capacity-bound
> prediction (mean D = -0.013 < tau, gate closed on 95/95 inputs) is a false positive;
> MK3 thus remains our only exemplar where empirical capacity-boundedness and the gate's
> capacity-bound classification coincide.

---
Run executed 2026-07-04 on GPU 3.
Command:
`.venv/bin/python experiments/scripts/longbench_gating.py --tasks passage_count --out experiments/results/longbench_passcount_qwen14b.jsonl --model Qwen/Qwen2.5-14B-Instruct --budgets 1.0,0.5,0.25,0.125,0.0625 --max_examples 100 --max_new 32 --gpu 3 --two_pass`
Analysis: `.venv/bin/python experiments/scripts/analyze_passcount.py experiments/results/longbench_passcount_qwen14b.jsonl`
