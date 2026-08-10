# Per-task decomposition of the 4x4 gating matrix (RULER 4K, tau=0.07)

## Convention: b < 1.0, tau = 0.07  (16 cells)

| cell | niah_multikey_3 | vt | fwe | qa_1 | all | no-MK3 | gate-open non-MK3 | gate-open MK3 |
|---|---|---|---|---|---|---|---|---|
| snapkv Qwen2.5-1.5B | +0.484 | +0.000 | +0.000 | +0.000 | +0.121 | +0.000 ** | 1.000 | 0.000 |
| snapkv Qwen2.5-3B | +0.734 | +0.004 | +0.300 | +0.000 | +0.259 | +0.101 | 0.653 | 0.000 |
| snapkv Qwen2.5-14B | +0.573 | +0.000 | +0.000 | +0.000 | +0.143 | +0.000 ** | 1.000 | 0.000 |
| snapkv Mistral-7B | +0.701 | +0.000 | +0.046 | +0.000 | +0.187 | +0.015 | 0.923 | 0.100 |
| h2o Qwen2.5-1.5B | +0.647 | +0.000 | +0.000 | +0.000 | +0.162 | +0.000 ** | 1.000 | 0.000 |
| h2o Qwen2.5-3B | +0.920 | +0.017 | +0.277 | +0.000 | +0.303 | +0.098 | 0.653 | 0.000 |
| h2o Qwen2.5-14B | +0.800 | +0.000 | +0.000 | +0.000 | +0.200 | +0.000 ** | 1.000 | 0.000 |
| h2o Mistral-7B | +0.883 | +0.000 | +0.043 | +0.017 | +0.236 | +0.020 | 0.923 | 0.100 |
| streamingllm Qwen2.5-1.5B | +0.650 | +0.000 | +0.000 | +0.000 | +0.163 | +0.000 ** | 1.000 | 0.000 |
| streamingllm Qwen2.5-3B | +0.930 | +0.030 | +0.760 | +0.000 | +0.430 | +0.263 | 0.653 | 0.000 |
| streamingllm Qwen2.5-14B | +1.000 | +0.000 | +0.000 | +0.000 | +0.250 | +0.000 ** | 1.000 | 0.000 |
| streamingllm Mistral-7B | +0.890 | +0.000 | +0.140 | +0.020 | +0.263 | +0.053 | 0.923 | 0.100 |
| pyramidkv Qwen2.5-1.5B | +0.610 | +0.000 | +0.000 | +0.000 | +0.152 | +0.000 ** | 1.000 | 0.000 |
| pyramidkv Qwen2.5-3B | +0.883 | +0.000 | +0.433 | +0.000 | +0.329 | +0.144 | 0.653 | 0.000 |
| pyramidkv Qwen2.5-14B | +0.913 | +0.000 | +0.000 | +0.000 | +0.228 | +0.000 ** | 1.000 | 0.000 |
| pyramidkv Mistral-7B | +0.863 | +0.000 | +0.060 | +0.000 | +0.231 | +0.020 | 0.923 | 0.100 |

- grand mean Delta, all four tasks: **+0.2286**
- grand mean Delta, excluding NIAH-MK3: **+0.0447**
- cells with non-MK3 Delta exactly 0.000: **8/16**
- in those cells the gate opens on 100% of non-MK3 inputs, so gated == plain by construction
- mean gate-open fraction: non-MK3 0.894, MK3 0.025

Arithmetically the matrix is Delta ~ (1/4) * Delta_MK3: mean Delta_MK3/4 = +0.1950 vs measured grand mean +0.2286.

## Budget-convention sensitivity (grand mean, all tasks)

- b < 1.0 (PINNED): +0.2286  (16 cells)
- all budgets incl. b = 1.0: +0.2286  (16 cells)
- b in {0.125, 0.25, 0.5}: +0.2402  (16 cells)
- 4 shared budgets {0.0625, 0.125, 0.25, 0.5}: +0.2431  (16 cells)
