# PRE-REGISTERED prediction: per-input recovery rate rho at RULER 32K (Qwen2.5-1.5B-Instruct)

Written: 2026-07-03 (UTC), AFTER the budget-1.0-only headroom run
(`ruler_32k_qwen15b_afull.jsonl`, done 2026-07-03T~02:43Z) and BEFORE any
eviction budget < 1.0 was run at 32K. The full sweep
(`ruler_32k_qwen15b_sweep.jsonl`) starts only after this file's sha256 is
logged in `scaling_32k_protocol.log`. Predictions below are frozen; only the
N note may be updated (if wall-time forces N=100 -> N=50) per protocol.

## Setup

- Model: Qwen/Qwen2.5-1.5B-Instruct, greedy decode, max_new 128.
- Data: SaylorTwift/RULER-32768-Qwen2.5-3B-tokenizer (official RULER
  generation at 32768 tokens under the Qwen2.5 tokenizer; simonjegou/ruler
  publishes no 32768 config). First 100 examples per task
  (vt, fwe, niah_multikey_3), same ordering as the headroom run.
- Pipeline: experiments/scripts/ruler_sweep_32k_wrapper.py ->
  ruler_sweep.py unchanged (SnapKV-style eviction, two-pass scoring,
  obs_window 32, n_sink 4, default 10-budget grid
  1.0,0.875,0.75,0.625,0.5,0.375,0.25,0.1875,0.125,0.0625).
- Correctness: reanalyze_ruler.py convention (vt/fwe: ALL golds substring
  in lowercased pred; niah_multikey_3: ANY).
- rho: fraction of the N inputs that are wrong at budget 1.0 and correct at
  >= 1 budget < 1.0 (same file, per-input).

## Step-1 fit (existing 4K/16K data only, recomputed and cross-checked
against paper Table tab:scaling-formula — all 8 audited cells match)

| task | cells (ratio rho/(1-A_full)) | mean | min-max band |
|---|---|---|---|
| VT  | 1.5B-4K 0.333, 1.5B-16K 0.253, 3B-16K 0.556 (3B-4K undefined, headroom 0) | 0.381 | [0.253, 0.556] |
| FWE | 1.5B-4K 0.077, 1.5B-16K 0.045, 3B-4K 0.042, 3B-16K 0.516 | 0.170 | [0.042, 0.516] |
| MK3 | 1.5B-4K 0.029, 1.5B-16K 0.000 | — | capacity-bound: rho ~ 0 |

Note (registered before measurement): the paper's scaling table draws its
FWE ratio cluster claim from the 3B-16K cell (0.52); the model-matched
1.5B FWE cells sit much lower (0.045-0.077). The primary FWE band uses all
four cells (min-max); a sharper secondary, model-matched (1.5B-only) band
[0.045, 0.077] is registered as exploratory.

## Measured headroom at 32K (budget 1.0 only, N=100/task)

| task | A_full | Wilson95 | 1 - A_full |
|---|---|---|---|
| vt              | 0.810 | [0.722, 0.875] | 0.190 |
| fwe             | 0.490 | [0.394, 0.587] | 0.510 |
| niah_multikey_3 | 0.140 | [0.085, 0.221] | 0.860 |

## PREDICTIONS (frozen)

Predicted rho interval = ratio band x (1 - A_full,32K); point = mean ratio x headroom.

| task | predicted rho interval (PRIMARY) | point | secondary (exploratory) |
|---|---|---|---|
| VT  | [0.048, 0.106] | 0.072 | — |
| FWE | [0.021, 0.263] | 0.087 | 1.5B-only band: [0.023, 0.039] |
| MK3 | rho <= 0.02 (protocol-fixed; despite headroom 0.860) | ~0 | — |

## Pre-registered verdict rules

1. Per task, PRIMARY verdict = measured rho point estimate inside the
   predicted interval above (MK3: rho <= 0.02). Wilson 95% CI of measured
   rho reported alongside for context.
2. Context-amplification check: the formula says rho tracks HEADROOM, not
   context length per se. At 32K the VT headroom (0.190) is far below its
   16K value (0.75) on this dataset, so the formula predicts VT rho FALLS
   from 0.19 (16K) to ~0.05-0.11 (32K) even though context doubled; FWE
   headroom shrinks 0.88 -> 0.51 (rho prediction ~0.02-0.26); MK3 headroom
   grows 0.74 -> 0.86 and rho must stay pinned at <= 0.02 for the
   capacity-bound side of the partition to hold.
3. N note: N=100 per task. (May be reduced to N=50 by wall-time rule
   BEFORE analysis; predictions do not change.)
