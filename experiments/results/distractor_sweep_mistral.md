# Distractor sweep: cross-architecture replication (Mistral-7B-Instruct-v0.3)

Mechanism-validation replication of tab:distractor (originally Qwen2.5-1.5B only).
Mistral-7B-Instruct-v0.3, RULER 4K, niah_multikey_{1,2,3}, N=50 per task.
D = mean early-layer-third head-agreement minus late-layer-third (Jaccard top-32 of
per-head attended key sets, obs_window=32). More layers/heads than Qwen-1.5B, so
absolute D differs; what matters is the monotone ordering MK1 > MK2 > MK3.

## Per-task mean D and per-input D range

| task | N | mean D | min D | max D |
|---|---:|---:|---:|---:|
| niah_multikey_1 | 50 | +0.0805 | +0.0423 | +0.1047 |
| niah_multikey_2 | 50 | +0.0715 | +0.0500 | +0.0971 |
| niah_multikey_3 | 50 | +0.0603 | +0.0401 | +0.0769 |

Qwen2.5-1.5B reference (tab:distractor): MK1 +0.1550, MK2 +0.1219, MK3 +0.0448.

## Monotonicity check

- MK1 > MK2 > MK3 on mean D: **True** (+0.0805 > +0.0715 > +0.0603)

## MK2 / MK3 per-input separation check

- MK2 range: [+0.0500, +0.0971]
- MK3 range: [+0.0401, +0.0769]
- Ranges separate (max_MK3 < min_MK2): **False** (+0.0769 < +0.0500)

## Verdict

The mechanism-validation gradient replicates cross-architecture: the early-to-late head-agreement drop D decreases monotonically as near-tie distractors are added (MK1 > MK2 > MK3) on Mistral-7B-Instruct-v0.3, matching the Qwen2.5-1.5B ordering, though the MK2/MK3 per-input ranges overlap.

