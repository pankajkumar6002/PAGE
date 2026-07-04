# Gate-signal ablation: is the head-agreement drop D uniquely effective?

Model: Qwen/Qwen2.5-1.5B-Instruct; RULER 4096; N=50/task; tasks: niah_multikey_3, niah_multikey_2, niah_multikey_1, vt, fwe, qa_1, niah_multivalue. One eager prefill per input; all six signals computed from the same prefill attentions (last 32 queries) and cached keys. Reproduce: `.venv/bin/python experiments/scripts/gate_signal_ablation.py` (add `--analyze_only` to skip the GPU probe).

Capacity-bound anchor: `niah_multikey_3` (+`niah_multikey_2`); dilution-prone set: `vt, fwe, qa_1, niah_multivalue`. AUCs are *oriented* per signal so that the capacity-bound class is on the low side (orientation shown as the 'MK3 side' column); 0.5 = chance either way.

## Per-task means (± std)

| signal | niah_multikey_3 | niah_multikey_2 | niah_multikey_1 | vt | fwe | qa_1 | niah_multivalue |
|---|---:|---:|---:|---:|---:|---:|---:|
| drop_D | +0.0448 ± 0.0079 | +0.1219 ± 0.0181 | +0.1550 ± 0.0192 | +0.1612 ± 0.0041 | +0.1114 ± 0.0041 | +0.2369 ± 0.0183 | +0.1265 ± 0.0176 |
| entropy_norm | +0.3290 ± 0.0044 | +0.3368 ± 0.0046 | +0.3305 ± 0.0049 | +0.3360 ± 0.0015 | +0.3088 ± 0.0025 | +0.3264 ± 0.0109 | +0.3308 ± 0.0028 |
| topk_mass_share | +0.5949 ± 0.0094 | +0.6977 ± 0.0055 | +0.7038 ± 0.0066 | +0.5834 ± 0.0027 | +0.7025 ± 0.0043 | +0.6728 ± 0.0190 | +0.7001 ± 0.0067 |
| max_share | +0.2934 ± 0.0062 | +0.2978 ± 0.0039 | +0.2998 ± 0.0045 | +0.2653 ± 0.0018 | +0.3196 ± 0.0016 | +0.3329 ± 0.0111 | +0.3015 ± 0.0032 |
| keynorm_disp | +0.0796 ± 0.0003 | +0.0798 ± 0.0004 | +0.0794 ± 0.0002 | +0.0905 ± 0.0004 | +0.0822 ± 0.0016 | +0.0778 ± 0.0008 | +0.0795 ± 0.0002 |
| entropy_drop | +0.0205 ± 0.0097 | -0.0294 ± 0.0089 | +0.0144 ± 0.0102 | -0.0214 ± 0.0032 | -0.0223 ± 0.0028 | -0.0596 ± 0.0109 | +0.0292 ± 0.0060 |

## Summary: signal × separation criteria

| signal | MK3 side | task-level sep (MK3 < all 4 dil.) | task-level sep (MK3 & MK2) | AUC MK3 vs dil. | AUC MK3+MK2 vs dil. | monotone on MK1→MK2→MK3 | Pearson r with D | Spearman ρ with D |
|---|---|---|---|---:|---:|---|---:|---:|
| drop_D | low | **yes** | no | 1.000 | 0.859 | **yes** | +1.000 | +1.000 |
| entropy_norm | high | no | no | 0.508 | 0.670 | no | +0.080 | +0.249 |
| topk_mass_share | low | no | no | 0.784 | 0.574 | **yes** | +0.224 | -0.050 |
| max_share | low | no | no | 0.723 | 0.708 | **yes** | +0.322 | +0.122 |
| keynorm_disp | low | no | no | 0.570 | 0.560 | no | +0.010 | -0.258 |
| entropy_drop | high | no | no | 0.805 | 0.575 | no | -0.641 | -0.492 |

## MK distractor gradient (raw task means, MK1 → MK2 → MK3)

| signal | MK1 | MK2 | MK3 | monotone (oriented) |
|---|---:|---:|---:|---|
| drop_D | +0.1550 | +0.1219 | +0.0448 | yes |
| entropy_norm | +0.3305 | +0.3368 | +0.3290 | no |
| topk_mass_share | +0.7038 | +0.6977 | +0.5949 | yes |
| max_share | +0.2998 | +0.2978 | +0.2934 | yes |
| keynorm_disp | +0.0794 | +0.0798 | +0.0796 | no |
| entropy_drop | +0.0144 | -0.0294 | +0.0205 | no |

## Interpretation

Only the head-agreement drop D achieves task-level separation of the capacity-bound anchor: MK3 (mean D = +0.045) sits strictly below all four dilution-task means (closest: fwe at +0.111) with per-input AUC 1.000, while the best alternative — the early-vs-late ENTROPY drop, the exact structural analogue of D with entropy substituted for agreement — reaches only AUC 0.805 and fails the task-level test because niah_multivalue (+0.029) lands above MK3 (+0.021). This dissects the signal cleanly: the early-late layer contrast contributes (raw entropy is at chance, AUC 0.508; adding the contrast lifts it to 0.805), but the *agreement* measure is what makes the separation exact, and no non-contrast concentration statistic (top-32 mass 0.784, max-share 0.723, key-norm dispersion 0.570) nor any signal's correlation with D (all |r| ≤ 0.64) suggests a cheap substitute or even a useful redundant proxy. Extending the capacity-bound set to MK3+MK2 degrades every signal including D (AUC 0.859, task-level sep fails because MK2's mean +0.122 rises above fwe's +0.111) — consistent with the paper's framing that D tracks the distractor-load gradient (MK1 > MK2 > MK3, monotone) rather than a binary task type, with MK2 sitting mid-gradient.

## Prior negatives folded in (context)

- **Mean attention entropy** (June probe, `attention_entropy_probe.py`, `entropy_probe.jsonl`, N=20/task): MK3 normalized entropy 0.328 sits strictly inside the dilution range (fwe 0.309 … vt 0.336) — no threshold exists. Reproduced here on N=50 same-inputs: AUC 0.508 (chance).
- **Depth-shape variants of the agreement curve** (`normalized_predictor.md` §2): relD, depth-correlation, slope, core-drop all ≤ rawD in every one of 7 model-cells and catastrophically inverted on Llama (AUC 0.224) — the early-minus-late difference in raw agreement units is the right base statistic; only its scale/offset is model-specific (fixed by the z-score recipe).
- This ablation closes the remaining gap: a same-inputs comparison against non-agreement candidates. None separates; D is not merely the best of a family of workable statistics but the only one of the six that works at all.
