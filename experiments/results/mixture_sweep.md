# Mixture-fraction sweep: Delta as a function of the NIAH-MK3 share f

Delta(f) = f * Delta_MK3 + (1 - f) * Delta_rest, per cell, then averaged.

| f (MK3 share) | 0.00 | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|---|---|---|---|
| grand mean Delta | +0.045 | +0.081 | +0.118 | +0.155 | +0.192 | +0.229 | +0.412 | +0.596 | +0.780 |

- Delta(f=0) = +0.0447, the no-MK3 grand mean
- Delta(f=0.25) = +0.2286, the paper's mixed suite and its +22.9pp headline
- slope between them: +0.7354 per unit MK3 share
- mean Delta_MK3 = +0.7801, mean Delta_rest = +0.0447

Reading: the headline number is a statement about a workload that is one quarter capacity-bound. On a workload with no capacity-bound inputs the gate is inert and Delta is near zero, which is the intended behaviour of a safeguard, not a failure of it.

## Per-cell endpoints

| cell | Delta_MK3 | Delta_rest |
|---|---:|---:|
| snapkv Qwen2.5-1.5B | +0.484 | +0.000 |
| snapkv Qwen2.5-3B | +0.734 | +0.101 |
| snapkv Qwen2.5-14B | +0.573 | +0.000 |
| snapkv Mistral-7B | +0.701 | +0.015 |
| h2o Qwen2.5-1.5B | +0.647 | +0.000 |
| h2o Qwen2.5-3B | +0.920 | +0.098 |
| h2o Qwen2.5-14B | +0.800 | +0.000 |
| h2o Mistral-7B | +0.883 | +0.020 |
| streamingllm Qwen2.5-1.5B | +0.650 | +0.000 |
| streamingllm Qwen2.5-3B | +0.930 | +0.263 |
| streamingllm Qwen2.5-14B | +1.000 | +0.000 |
| streamingllm Mistral-7B | +0.890 | +0.053 |
| pyramidkv Qwen2.5-1.5B | +0.610 | +0.000 |
| pyramidkv Qwen2.5-3B | +0.883 | +0.144 |
| pyramidkv Qwen2.5-14B | +0.913 | +0.000 |
| pyramidkv Mistral-7B | +0.863 | +0.020 |
