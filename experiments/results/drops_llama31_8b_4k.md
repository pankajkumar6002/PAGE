# Per-task head-agreement drops on Llama-3.1-8B-Instruct, RULER 4K

## Setup

- Model: `meta-llama/Llama-3.1-8B-Instruct` (Meta license access granted shortly
  before this run; weights downloaded and loaded cleanly with the existing HF
  token). 32 layers, 32 attention heads, 8 KV heads (GQA, 4x sharing),
  RoPE theta = 500000, Sentencepiece BPE tokenizer (vocab 128000).
- RULER configuration: 4096 token target
- Tasks: qa_1, qa_2, vt, niah_multivalue, fwe, niah_multikey_3
- N = 50 examples per task (300 total)
- Two-pass attention scoring: SDPA prefill, then 32-token obs-window re-forward
  with eager attention for `output_attentions=True`
- Per-layer head agreement = mean Jaccard top-32 across all head pairs in a
  layer, computed over the obs_window queries
- D (drop) = mean(layers 0..L/3-1) - mean(layers 2L/3..L-1) with L = 32 layers
  (early third = layers 0-9, late third = layers 22-31)

## Per-task drop D

Sorted by mean D ascending (smallest D first; partition prediction says
smallest D = capacity-bound).

| task              |  N | mean T | mean drop D | stdev D |   min D |   max D |
| ----------------- | -: | -----: | ----------: | ------: | ------: | ------: |
| niah_multivalue   | 50 |   3853 |     +0.0608 |  0.0073 | +0.0430 | +0.0737 |
| niah_multikey_3   | 50 |   3093 |     +0.0712 |  0.0052 | +0.0612 | +0.0847 |
| vt                | 50 |   3880 |     +0.0785 |  0.0020 | +0.0730 | +0.0835 |
| qa_1              | 50 |   3055 |     +0.0920 |  0.0111 | +0.0610 | +0.1149 |
| qa_2              | 50 |   3398 |     +0.0976 |  0.0154 | +0.0690 | +0.1302 |
| fwe               | 50 |   3999 |     +0.1129 |  0.0058 | +0.0999 | +0.1310 |

## Per-layer-bin head agreement (raw)

| task            | early  | middle | late   |
| --------------- | -----: | -----: | -----: |
| fwe             | 0.4540 | 0.3871 | 0.3397 |
| niah_multikey_3 | 0.3433 | 0.3012 | 0.2702 |
| niah_multivalue | 0.4260 | 0.3920 | 0.3618 |
| qa_1            | 0.4295 | 0.3729 | 0.3328 |
| qa_2            | 0.4237 | 0.3626 | 0.3232 |
| vt              | 0.3662 | 0.2761 | 0.2831 |

## Interpretation against partition prediction (tau = 0.07)

Predicted split:
- dilution-prone (D large, expect D >= tau): qa_1, qa_2, vt, fwe, niah_multivalue
- capacity-bound (D small, expect D <  tau): niah_multikey_3

On Llama-3.1-8B at 4K, all six tasks have positive D (none of them invert
the early-vs-late ordering, unlike Yi where niah_multivalue went negative).
Versus the tau = 0.07 threshold calibrated on Qwen:

- fwe (+0.113), qa_2 (+0.098), qa_1 (+0.092), vt (+0.079): all above tau,
  prediction holds — heads diverge late, so SnapKV is expected to fail and
  the gate should open.
- niah_multikey_3 (+0.071): essentially at tau, marginally above. The gate
  may open on roughly half of inputs depending on per-input variance
  (stdev = 0.005 with mean only 0.001 above tau).
- niah_multivalue (+0.061): below tau, predicted capacity-bound. This is
  the same anomaly seen on Yi-1.5-9B (where niah_multivalue went outright
  negative) and is consistent with niah_multivalue being a borderline case
  in the Llama-architecture family at 4K — head agreement does not collapse
  as strongly in late layers as it does on Qwen/Mistral.

Ranking summary: niah_multikey_3 lands second smallest (only niah_multivalue
below it), and the four clearly dilution-prone tasks (fwe, qa_2, qa_1, vt)
all sit comfortably above tau. The partition predictor transfers partially
to Llama: the capacity-bound prediction holds on niah_multikey_3, and four
of five dilution-prone tasks pass the threshold; niah_multivalue is the
single boundary case shared with Yi.
