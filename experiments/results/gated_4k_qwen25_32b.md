# Qwen2.5-32B-Instruct 4K gated row (scale validation)

Setup: Qwen2.5-32B-Instruct, RULER 4K mixed suite (niah_multikey_3, vt, fwe,
qa_1), N=100/task, budgets {1.0, 0.5, 0.25, 0.125, 0.0625}, SnapKV,
two-pass SDPA prefill (mandatory: 62GB bf16 weights). tau=0.07.
Raw: gated_4k_qwen25_32b.jsonl (2000 records, complete coverage).

## Full-KV accuracy: the 4K/32B saturation regime

| task | A_full |
|---|---:|
| niah_multikey_3 | 1.000 |
| vt | 1.000 |
| qa_1 | 0.940 |
| fwe | 0.930 |

At 32B and 4K the model is near-saturated (A_full 0.93-1.00). Per the paper's
scaling formula rho ~ (1 - A_full) x dilution, this predicts rho ~ 0 (almost no
dilution headroom to recover) -- the model-size-saturation regime. So the
gate's demonstrated value here is capacity-bound PROTECTION, not dilution
recovery.

## Partition ordering: transfers

| task | mean drop |
|---|---:|
| niah_multikey_3 | +0.025 |
| fwe | +0.052 |
| vt | +0.079 |
| qa_1 | +0.103 |

NIAH-MK3 is the smallest drop -- the ordering scales to 32B.

## Gated results (tau=0.07)

| task | plain | gated | Delta | plain @ b=0.0625 | gate |
|---|---:|---:|---:|---:|---|
| niah_multikey_3 | 0.117 | 1.000 | +0.882 | 0.00 | closed |
| fwe | 0.492 | 0.930 | +0.438 | 0.09 | closed |
| vt  | 0.667 | 0.688 | +0.020 | 0.00 | open (0.94) |
| qa_1 | 0.915 | 0.915 | +0.000 | 0.87 | open (0.99) |
| **mean** | **0.548** | **0.883** | **+0.335** | | kept 0.63 |

z-scored threshold: pooled mu=0.065, sd=0.030, tau_z=mu-0.69sd=0.044.
At tau_z: Delta=+0.237, mean kept-KV 0.447 (more compression), gate opens
fwe/vt/qa_1, closed MK3.

## Verdict

1. **Partition ordering scales to 32B** (MK3 smallest drop).
2. **32B/4K is the saturation regime the scaling formula predicts** (A_full
   0.93-1.00, near-zero dilution headroom). The gate's value here is
   capacity-bound protection: plain SnapKV collapses at low budgets (MK3 0.00,
   fwe 0.09, vt 0.00 at b=0.0625) while the gate holds full-KV on the low-drop
   tasks, giving +0.335 mean (driven by MK3 +0.88 and fwe +0.44).
3. z-scoring trades some Delta for more compression (kept 0.63 -> 0.45), same
   as on other families.

Honest note: the large mean Delta at 32B is protection-dominated, not
dilution-recovery -- exactly what the model-size-saturation arm of the scaling
formula predicts for a strong model at short context. It is a scale-validation
point for the partition and the predictor, not additional dilution evidence.
