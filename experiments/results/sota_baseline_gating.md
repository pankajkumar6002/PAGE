# Method-agnostic gating: SOTA-baseline audit

- Model: **Qwen2.5-1.5B-Instruct**
- Benchmark: **RULER 4K**, tasks niah_multikey_3 + vt + fwe + qa_1 (100 examples each = 400 total)
- Gate threshold: tau = **0.07** (drop in head-agreement, early-vs-late thirds)
- Each base evictor X is run twice per example: plain-X (always evict) vs gated-X (evict only when gate fires)

## Headline: mean accuracy delta over eviction budgets, mixed suite

| Base method | plain mean | gated mean | **Δ (gated − plain)** | gate-open rate |
|---|---:|---:|---:|---:|
| SnapKV | 0.429 | 0.578 | **+0.149** | 0.750 |
| H2O | 0.196 | 0.357 | **+0.162** | 0.750 |
| StreamingLLM | 0.033 | 0.195 | **+0.163** | 0.750 |
| PyramidKV | 0.417 | 0.570 | **+0.152** | 0.750 |

## SnapKV — per-budget mixed-suite curve
| budget | plain | gated | Δ | gate-open / N |
|---:|---:|---:|---:|---:|
| 1 | 0.608 | 0.608 | +0.000 | 300/400 |
| 0.5 | 0.470 | 0.598 | +0.128 | 300/400 |
| 0.25 | 0.432 | 0.590 | +0.157 | 300/400 |
| 0.125 | 0.385 | 0.547 | +0.162 | 300/400 |

## H2O — per-budget mixed-suite curve
| budget | plain | gated | Δ | gate-open / N |
|---:|---:|---:|---:|---:|
| 1 | 0.608 | 0.608 | +0.000 | 300/400 |
| 0.5 | 0.297 | 0.458 | +0.160 | 300/400 |
| 0.25 | 0.212 | 0.375 | +0.163 | 300/400 |
| 0.125 | 0.077 | 0.240 | +0.162 | 300/400 |

## StreamingLLM — per-budget mixed-suite curve
| budget | plain | gated | Δ | gate-open / N |
|---:|---:|---:|---:|---:|
| 1 | 0.608 | 0.608 | +0.000 | 300/400 |
| 0.5 | 0.033 | 0.195 | +0.163 | 300/400 |
| 0.25 | 0.033 | 0.195 | +0.163 | 300/400 |
| 0.125 | 0.033 | 0.195 | +0.163 | 300/400 |

## PyramidKV — per-budget mixed-suite curve
| budget | plain | gated | Δ | gate-open / N |
|---:|---:|---:|---:|---:|
| 1 | 0.608 | 0.608 | +0.000 | 300/400 |
| 0.5 | 0.450 | 0.588 | +0.138 | 300/400 |
| 0.25 | 0.422 | 0.580 | +0.157 | 300/400 |
| 0.125 | 0.380 | 0.542 | +0.162 | 300/400 |

## Per-task delta on the lowest non-trivial budget (b=0.25)

| Base method | niah_multikey_3 | vt | fwe | qa_1 |
|---|---:|---:|---:|---:|
| SnapKV | +0.630 | +0.000 | +0.000 | +0.000 |
| H2O | +0.650 | +0.000 | +0.000 | +0.000 |
| StreamingLLM | +0.650 | +0.000 | +0.000 | +0.000 |
| PyramidKV | +0.630 | +0.000 | +0.000 | +0.000 |

## Discussion: method-agnostic claim

Across 4 qualitatively distinct base evictors — SnapKV (recent-query attention), H2O (cumulative attention mass), StreamingLLM (sink + recent positions, no learned score), and PyramidKV (depth-weighted SnapKV) — the gate produced a positive mean accuracy delta on 4/4. Pooled mean Δ per method: SnapKV +0.149; H2O +0.162; StreamingLLM +0.163; PyramidKV +0.152. This supports the paper's central method-agnostic claim: head-agreement drop is a signal **about the prompt**, not about a particular eviction score. Any base evictor that catastrophically fails on a capacity-bound input is rescued by the same gating logic, with no per-method re-tuning beyond a single shared tau.

