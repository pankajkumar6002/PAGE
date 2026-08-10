# b = 1.0 identity audit

At b=1.0 the gated arm must keep the full cache and match the plain outcome row for row. If it does, r(x) >= 0 is an identity and the empty r=-1 bin is vacuous.

| cell | rows at b=1.0 | n_kept == T | gated == plain | gate closed |
|---|---:|---:|---:|---:|
| snapkv Qwen2.5-1.5B 4K | 400 | 400/400 | 400/400 | 100/400 |
| snapkv Qwen2.5-3B 4K | 400 | 400/400 | 400/400 | 172/400 |
| snapkv Qwen2.5-14B 4K | 200 | 200/200 | 200/200 | 50/200 |
| snapkv Mistral-7B 4K | 400 | 400/400 | 400/400 | 113/400 |
| h2o Qwen2.5-1.5B 4K | 400 | 400/400 | 400/400 | 100/400 |
| h2o Qwen2.5-3B 4K | 400 | 400/400 | 400/400 | 204/400 |
| h2o Qwen2.5-14B 4K | 200 | 200/200 | 200/200 | 50/200 |
| h2o Mistral-7B 4K | 400 | 400/400 | 400/400 | 113/400 |
| streamingllm Qwen2.5-1.5B 4K | 400 | 400/400 | 400/400 | 100/400 |
| streamingllm Qwen2.5-3B 4K | 400 | 400/400 | 400/400 | 204/400 |
| streamingllm Qwen2.5-14B 4K | 200 | 200/200 | 200/200 | 50/200 |
| streamingllm Mistral-7B 4K | 400 | 400/400 | 400/400 | 113/400 |
| pyramidkv Qwen2.5-1.5B 4K | 400 | 400/400 | 400/400 | 100/400 |
| pyramidkv Qwen2.5-3B 4K | 400 | 400/400 | 400/400 | 204/400 |
| pyramidkv Qwen2.5-14B 4K | 200 | 200/200 | 200/200 | 50/200 |
| pyramidkv Mistral-7B 4K | 400 | 400/400 | 400/400 | 113/400 |
| 16k_qwen15b_sdpa | 200 | 200/200 | 200/200 | 50/200 |
| 16k_qwen3b | 200 | 200/200 | 200/200 | 135/200 |
| 16k_qwen14b | 120 | 120/120 | 120/120 | 62/120 |
| 16k_mistral7b | 200 | 200/200 | 200/200 | 47/200 |

Cells audited: 20

CHECK: PASS - r(x) >= 0 holds by construction on every audited cell, so the empty r=-1 bin is vacuous and cannot be cited as evidence of harmlessness.
