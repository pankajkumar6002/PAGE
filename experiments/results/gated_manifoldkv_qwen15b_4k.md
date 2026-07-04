# Gated ManifoldKV-variant composition test (2026-07-02/03)

Setup: Qwen2.5-1.5B-Instruct, RULER 4K mixed suite (niah_multikey_3, vt, fwe,
qa_1), N=100/task, budgets {1.0, 0.5, 0.25, 0.125, 0.0625}, tau=0.07,
score_policy=manifoldkv (new in gated_eviction.py). Run: 2401s on GPU 2,
gate_open_frac 0.75. Raw: gated_manifoldkv_qwen15b_4k.jsonl.

Scorer: per-(layer, kv-head) L2 distance of each key to the head's key
centroid, averaged into ONE [T] score shared across layers (single-mask
constraint of this codebase; the ManifoldKV paper evicts per layer/head).
Higher distance = keep. Same sink/recency protection as all other policies.
The GATE signal (head-agreement drop) is attention-based and unchanged.

## Results (plain / gated accuracy per budget)

| task | b=0.5 | b=0.25 | b=0.125 | b=0.0625 | full-KV |
|---|---|---|---|---|---|
| niah_multikey_3 | 0.01 / **0.65** | 0.00 / **0.65** | 0.00 / **0.65** | 0.00 / **0.65** | 0.65 |
| vt  | 0.34 / 0.34 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.82 |
| fwe | 0.34 / 0.34 | 0.16 / 0.16 | 0.00 / 0.00 | 0.00 / 0.00 | 0.22 |
| qa_1 | 0.68 / 0.68 | 0.47 / 0.47 | 0.29 / 0.29 | 0.21 / 0.21 | 0.74 |

Mean over 16 (task, budget) cells: plain 0.156, gated 0.318, **Δ = +16.2pp**.
Gated ≥ plain in every cell (equal where the gate opens, by construction).

## Verdicts

(i) **MK3 rescue: NOT reproduced by our variant.** The single-mask,
layer-averaged Euclidean-outlier scorer collapses on MK3 exactly like the
attention-score evictors (0.01 at b=0.5). This does NOT refute ManifoldKV's
reported 92.4% — their design is per-layer/per-head and integrates with
Ada-KV; the paper's capacity-bound scoping should keep citing their numbers
and note our variant did not reproduce the rescue.

(ii) **The wrapper composes with a geometry-based scorer.** Same gate, same
tau, fifth base evictor, Δ = +16.2pp, no negative cell, MK3 +65pp at every
budget. The gate signal is scorer-independent by construction and this
confirms it empirically outside the attention-score family.

(iii) Notable: as a plain evictor the single-mask geometric scorer is much
weaker than SnapKV on dilution-prone tasks (vt 0.34 vs ~0.75 at b=0.5) —
retaining geometric outliers keeps the wrong tokens for aggregation/tracking.
