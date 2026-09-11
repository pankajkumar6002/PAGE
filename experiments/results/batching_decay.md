# Expected compression vs batch size

Under a **static** allocator a batch is provisioned for its largest
resident cache, so one gate-closed sequence erases the benefit for the
whole batch:

    E[kept](B) = p_open^B * k_open + (1 - p_open^B) * 1.0

`p_open` is measured per cell at tau = 0.07; `k_open` is the realized
kept-KV fraction at the aggressive budget b = 0.0625 (nominal 16x).

| cell | p_open | k_open | B=1 | B=2 | B=4 | B=8 | B=16 | B=32 | B=64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B | 0.750 | 0.062 | 3.37x | 2.12x | 1.42x | 1.10x | 1.01x | 1.00x | 1.00x |
| Qwen2.5-3B | 0.490 | 0.062 | 1.85x | 1.29x | 1.06x | 1.00x | 1.00x | 1.00x | 1.00x |
| Qwen2.5-14B | 0.750 | 0.062 | 3.37x | 2.12x | 1.42x | 1.10x | 1.01x | 1.00x | 1.00x |
| Mistral-7B | 0.718 | 0.062 | 3.06x | 1.93x | 1.33x | 1.07x | 1.00x | 1.00x | 1.00x |
| **mean of per-cell compressions** | (0.677) | (0.062) | **2.91x** | **1.86x** | **1.31x** | **1.07x** | **1.01x** | **1.00x** | **1.00x** |

The p_open and k_open shown on the mean row are averages of the inputs, given for reference only; the compressions are the mean of the per-cell compressions, since the model is nonlinear in p_open.

## What this settles

At the measured mean p_open = 0.68, the probability that every
sequence in a batch of 32 opens is 3.8e-06, so expected
compression is 1.00x. **For a static batch-serving system the
memory benefit is close to zero.**

The paper should state this scope in the abstract rather than leaving
the mechanism unquantified in Section 5. PAGE is a single-stream or
small-batch safeguard under static provisioning.

## Scope of the model

This is a static max-over-batch allocator. **Continuous batching** with
per-sequence paged allocation (PagedAttention/vLLM) does not have this
property: it pages per sequence, so a gate-closed sequence costs its own
pages rather than the batch's. The table above is therefore an upper
bound on the damage, and the paper must say which regime it claims.
The cost that does NOT go away under paging is the extra
attention-materializing prefill pass, which is paid on 100% of requests
to save cache on the fraction where the gate closes.

## Checks

- [PASS] compression monotonically decreasing in B (got ok)
- [PASS] mean compression at B=32 == 1.00 (got 1.0001)
- [PASS] Qwen2.5-1.5B: B=1 compression > 1x (got 3.37x (N=400))
- [PASS] Qwen2.5-3B: B=1 compression > 1x (got 1.85x (N=400))
- [PASS] Qwen2.5-14B: B=1 compression > 1x (got 3.37x (N=200))
- [PASS] Mistral-7B: B=1 compression > 1x (got 3.06x (N=400))
