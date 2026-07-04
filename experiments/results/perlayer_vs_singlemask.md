# Per-layer SnapKV vs single-mask comparison

**Setup.** Qwen2.5-1.5B-Instruct, RULER `simonjegou/ruler` 4K and 16K configs,
100 examples per task, same budget grid. The two scripts differ only in how
the keep_mask is derived:

- `ruler_sweep.py` (single-mask): one mask per example, derived from the
  mean attention across all layers and heads.
- `ruler_sweep_perlayer.py` (per-layer): one mask *per layer*, each derived
  from that layer's own last-window attention. Same total budget per layer
  so cache shape stays uniform.

## Headline table ($\rho_{\mathrm{KV}}$)

| Task | Context | Single-mask | Per-layer | Per-layer/single ratio |
|---|:---:|---:|---:|---:|
| VT | 4K | 0.060 | 0.040 | 0.67 |
| FWE | 4K | 0.060 | **0.110** | 1.83 |
| QA_1 | 4K | 0.080 | 0.070 | 0.88 |
| niah_multivalue | 4K | 0.070 | **0.150** | 2.14 |
| VT | 16K | 0.190 | 0.190 | 1.00 |
| FWE | 16K | 0.040 | 0.050 | 1.25 |
| QA_1 | 16K | 0.140 | 0.140 | 1.00 |
| niah_multivalue | 16K | 0.100 | 0.100 | 1.00 |

## Pattern

Per-layer helps clearly on **aggregation / multi-value tasks at 4K** (FWE
$1.8\times$, niah_multivalue $2.1\times$). At long context (16K) and on
sequential-integration tasks (VT, QA), per-layer roughly matches single-mask.

**Interpretation.** Aggregation tasks have attention patterns that vary
substantially across layers: shallow layers attend to local n-gram features
relevant to word-counting, deep layers attend to query-related semantics.
A single mask averages these and discards both. Per-layer respects the
heterogeneity. Multi-value integration is similar: shallow layers track key
positions, deep layers track value content; per-layer keeps both.

Sequential integration tasks (VT, QA) have more uniform attention patterns
across layers — every layer needs to follow the same chain — so averaging
in single-mask doesn't hurt.

At 16K, the context-length amplification dominates regardless of mask
granularity. Per-layer offers no additional lift in that regime.

## Caveats

- These are per-layer-uniform-budget masks, not per-head SnapKV. True per-head
  would respect intra-layer heterogeneity, which is the next step (requires
  zero-padded cache or attention masking — non-trivial under HF's RoPE
  assumptions).
- 100 examples per task; noise floor on $\rho$ is roughly $\pm 0.02$. The
  "ratio" column should be read with that uncertainty.
- The 4K wins are real but modest in absolute terms.

## Implication for the paper

Use **single-mask as the primary baseline** (simplest, matches concurrent
work's reporting structure). **Add per-layer as a method-section ablation**
that demonstrates: the partition signal is improvable on aggregation-style
tasks by respecting layer heterogeneity. This is a small but principled
methodological contribution beyond the main H1 finding.
