# One-shot τ calibration recipe

## The recipe

For a new (model, context) cell, calibrate τ from a 50-input pilot:

1. Pick 20 inputs from a known capacity-bound task (NIAH-MultiKey-3 in RULER works
   as the canonical anchor).
2. Pick 20 inputs from a known dilution-prone task (VT in RULER, or any
   multi-hop / aggregation task).
3. Run the prefill + head-agreement-drop computation on each (≈ one prefill cost
   per input, no labels needed).
4. Set
   $$\tau = \frac{\bar{D}_{\mathrm{NIAH\mbox{-}MK3}} + \bar{D}_{\mathrm{VT}}}{2}.$$

The recipe replaces our hand-set $\tau = 0.07$ with the cell-specific midpoint.
This is a single-figure-of-merit decision on a 50-input pilot — no
backward pass, no labels, no training.

## Empirical validation (4 cells, post-hoc)

We use existing data: per-layer-agreement probes give the two drops; gated
eviction runs give the gated-mean-accuracy as a function of τ via post-hoc
re-evaluation (saved `correct_plain` at b=1.0 is the full-KV outcome).

| Cell | $\bar{D}_{\mathrm{NIAH\mbox{-}MK3}}$ | $\bar{D}_{\mathrm{VT}}$ | $\tau_{\mathrm{calibrated}}$ | gated@τ_cal | gated@τ=0.07 | plain | best |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen 1.5B 4K | +0.045 | +0.161 | +0.103 | 0.559 (+12.1pp) | 0.559 (+12.1pp) | 0.438 | tied |
| Qwen 3B 4K | -0.002 | +0.076 | +0.037 | 0.792 (+21.3pp) | 0.839 (+25.9pp) | 0.580 | **τ=0.07** |
| **Qwen 3B 16K** | -0.021 | +0.068 | +0.024 | 0.584 (+5.4pp) | 0.564 (+3.4pp) | 0.530 | **τ_cal** |
| **Mistral 4K** | +0.060 | +0.096 | +0.078 | 0.845 (+22.3pp) | 0.810 (+18.7pp) | 0.623 | **τ_cal** |

## What this shows

- **The recipe fixes the Qwen 3B 16K cell** (the only failure case for τ=0.07), lifting gating Δ from +3.4pp to +5.4pp.
- **The recipe improves Mistral 4K** by +3.5pp.
- **The recipe slightly hurts Qwen 3B 4K** by -4.5pp (gating gives up some of its margin).
- **No-op on Qwen 1.5B 4K** (both τ values give identical gated accuracy).

Average across the 4 cells: τ_calibrated = +15.3pp Δ vs τ=0.07 = +15.0pp Δ —
recipe is essentially neutral on average but turns the failure case into a
moderate win.

## What the recipe costs

- One prefill per pilot input (~40 examples total).
- One layer-bin head agreement Jaccard top-32 computation per example
  (~50ms on Qwen 1.5B at 4K, see `gated_eviction.py` head_agreement_layer).
- No labels, no training, no backward pass.

Total: under 5 minutes of inference on a single A100 for any cell tested.

## Position in the paper

This recipe addresses reviewer objection O3 (per `paper/strategic_analysis.md`):
"the predictor doesn't transfer across architectures." The answer: the bare
drop is a robust *ordering statistic* in every cell tested; the absolute
threshold is the only thing that needs light per-cell calibration, and the
calibration is a 50-input no-label pilot.

We recommend reporting both numbers in the paper (gated@τ=0.07 *and*
gated@τ_calibrated) so readers see (i) the simplicity of the fixed
threshold and (ii) the robustness of the recipe.
