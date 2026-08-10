# Task-label oracle vs PAGE (RULER 4K matrix, tau = 0.07)

Oracle: close the gate on every NIAH-MK3 input, open on every other. It is not implementable, since it needs the task label PAGE never observes; it is the ceiling for any task-level policy.

| cell | PAGE $\Delta$ | oracle $\Delta$ | gap | gate-open on MK3 |
|---|---:|---:|---:|---:|
| snapkv Qwen2.5-1.5B | +0.121 | +0.121 | +0.000 | 0.000 |
| snapkv Qwen2.5-3B | +0.259 | +0.183 | +0.076 | 0.000 |
| snapkv Qwen2.5-14B | +0.143 | +0.143 | +0.000 | 0.000 |
| snapkv Mistral-7B | +0.187 | +0.195 | -0.008 | 0.100 |
| h2o Qwen2.5-1.5B | +0.162 | +0.162 | +0.000 | 0.000 |
| h2o Qwen2.5-3B | +0.303 | +0.230 | +0.073 | 0.000 |
| h2o Qwen2.5-14B | +0.200 | +0.200 | +0.000 | 0.000 |
| h2o Mistral-7B | +0.236 | +0.243 | -0.007 | 0.100 |
| streamingllm Qwen2.5-1.5B | +0.163 | +0.163 | +0.000 | 0.000 |
| streamingllm Qwen2.5-3B | +0.430 | +0.232 | +0.198 | 0.000 |
| streamingllm Qwen2.5-14B | +0.250 | +0.250 | +0.000 | 0.000 |
| streamingllm Mistral-7B | +0.263 | +0.247 | +0.015 | 0.100 |
| pyramidkv Qwen2.5-1.5B | +0.152 | +0.152 | +0.000 | 0.000 |
| pyramidkv Qwen2.5-3B | +0.329 | +0.221 | +0.108 | 0.000 |
| pyramidkv Qwen2.5-14B | +0.228 | +0.228 | +0.000 | 0.000 |
| pyramidkv Mistral-7B | +0.231 | +0.240 | -0.009 | 0.100 |
| **grand mean** | **+0.229** | **+0.201** | **+0.028** | |

- PAGE reproduces the oracle exactly in **8 of 16** cells. There the gate is a perfect task classifier and D buys label-freeness, not resolution.
- PAGE **beats** the oracle in **5** cells, by up to +0.198. A task-level policy cannot do this: the gain comes from closing on individual dilution-prone inputs the oracle opens.
- PAGE **loses** in **3** cells, by at most -0.009, from false positives on MK3.

Honest summary: on this suite most of the matrix Delta is reproducible by a task-label oracle, so the headline is largely a task-level effect. The per-input claim rests on the cells where PAGE exceeds the oracle, and on the Llama-3.1-8B MK3 cell where mean D sits above tau yet per-input dispersion still closes the gate on the inputs that need it.
