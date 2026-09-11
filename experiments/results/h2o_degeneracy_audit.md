# The Qwen2.5-14B H2O column duplicates SnapKV

Joined on (task, id, budget) between `gated_4k_{slug}.jsonl` (SnapKV)
and `gated_{h2o}_{slug}_4k.jsonl`, **excluding b = 1.0** where both
arms are the full cache by construction.

| cell | rows | plain-outcome agreement | `drop` identical |
|---|---:|---:|---:|
| Qwen2.5-1.5B | 1200 | 0.7250 | 1.0000 |
| Qwen2.5-3B | 1200 | 0.7092 | 1.0000 |
| Qwen2.5-14B | 600 | 1.0000  <- duplicate | 1.0000 |
| Mistral-7B | 1200 | 0.7342 | 1.0000 |

## What this settles

Qwen2.5-14B H2O agrees with SnapKV on **100.0% of
600 rows**, against 71-73% for the other three
models. It is not a weakened independent evictor, it is the same
measurement reported twice.

**Fix in the paper:** dagger the cell in `tab:matrix` and add to the
caption: *Qwen2.5-14B ran two-pass prefill, where H2O's all-query
score reduces exactly to SnapKV's; this column duplicates the SnapKV
column (identical on 600/600 rows) and is not an
independent base evictor.* Report the matrix as **15 independent
cells**, and give the grand mean both ways.

## The good half of the same measurement

`drop` is identical to 1e-9 on **100% of rows in every model**. The gate
signal is computed from prefill attentions and does not depend on which
base evictor consumes it. That is the method-agnosticism claim, measured
rather than asserted, and it is worth quoting in the paper.

## Checks

- [PASS] Qwen2.5-14B H2O agreement == 1.0000 (got 1.0000)
- [PASS] Qwen2.5-1.5B H2O agreement < 0.8 (independent) (got 0.7250)
- [PASS] Qwen2.5-3B H2O agreement < 0.8 (independent) (got 0.7092)
- [PASS] Mistral-7B H2O agreement < 0.8 (independent) (got 0.7342)
- [PASS] `drop` identical on 100% of rows in every model (got ok)
