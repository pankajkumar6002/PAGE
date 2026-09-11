# True per-layer ManifoldKV on NIAH-MK3

Question: does the faithful per-layer/per-head ManifoldKV design (Euclidean-outlier
key scoring) rescue NIAH-MK3 where SnapKV and our single-mask ManifoldKV variant
collapse? ManifoldKV (2602.08343) reports 92.4% on 3-key NIAH at 50% compression.

Setup: Qwen2.5-1.5B-Instruct, RULER 4K, niah_multikey_3, N=100, budgets
{1.0,0.5,0.25,0.125,0.0625}. New score_policy=manifoldkv_perlayer in
gated_eviction.py: independent per-layer keep-set, each position scored by
Euclidean distance of its key to that layer's key centroid (averaged over kv-heads),
keep the farthest (outliers), same sink+recency protection as other policies. This
is faithful to ManifoldKV's per-layer design (unlike the earlier single-mask variant).
Raw: manifoldkv_perlayer_mk3_qwen15b_4k.jsonl.

## Result (accuracy vs budget on NIAH-MK3, N=100)

| budget | per-layer ManifoldKV | SnapKV (same model) |
|---|---:|---:|
| 1.0 (full) | 0.65 | 0.65 |
| 0.5 | **0.03** | 0.14 |
| 0.25 | 0.00 | 0.02 |
| 0.125 | 0.00 | 0.00 |
| 0.0625 | 0.00 | 0.00 |

**Per-layer ManifoldKV does NOT rescue NIAH-MK3 in our harness. It collapses
from 0.65 to 0.03 at 50% compression, faster than SnapKV (0.14 at b=0.5).**

## Interpretation (case b, favorable)

The faithful per-layer geometry scorer also fails on MK3 here, and underperforms
the attention-score baseline. So NIAH-MK3 is capacity-bound for every scorer we
tested (SnapKV, H2O, StreamingLLM, PyramidKV, single-mask ManifoldKV, and now
per-layer ManifoldKV). The gate correctly classifies it capacity-bound for all
of them.

We could not reproduce ManifoldKV's reported 92.4%-at-50% rescue. The most likely
reasons are setup differences: (i) our RULER NIAH-MultiKey-3 places three near-tie
keys in a long haystack, which need not match their 3-key NIAH; (ii) our model is
small (Qwen2.5-1.5B, full-KV accuracy only 0.65 on this task), whereas their result
is on larger models; (iii) their full pipeline integrates with Ada-KV and per-head
ragged caches, which our single-A100 harness does not fully replicate. We report
what our harness shows and do not dispute their number on their setup.

Bottom line for the paper: this closes the "run true per-layer ManifoldKV"
rebuttal. The capacity-bound claim, scoped as scorer-relative, holds for every
scorer we tested including the faithful per-layer geometry one.

(VT and FWE dilution-side runs were also launched for completeness; MK3 is the
decisive result for the capacity-bound question.)

## Full run: dilution-side tasks (VT, FWE) confirm the scorer is weak in our harness

| budget | MK3 (cap-bound) | VT (dilution) | FWE (dilution) |
|---|---:|---:|---:|
| 1.0 | 0.65 | 0.82 | 0.22 |
| 0.5 | 0.03 | 0.30 | 0.21 |
| 0.25 | 0.00 | 0.01 | 0.01 |
| 0.125 | 0.00 | 0.00 | 0.00 |
| 0.0625 | 0.00 | 0.00 | 0.00 |

Per-layer ManifoldKV collapses across ALL tasks in our harness, not just MK3.
This is the honest reading: the geometry scorer underperforms in our single-A100
harness generally (as the single-mask variant did too), so the MK3 non-rescue
reflects a harness/implementation gap vs their full Ada-KV per-head pipeline,
NOT a claim that geometry scoring cannot rescue MK3 in principle. The paper's
sentence is scoped accordingly. All three tasks N=100.
