# Achieved cache size, Pareto frontier, and oracle gap for the τ = 0.07 gate

Reproduce all numbers and the figure:
`.venv/bin/python experiments/scripts/pareto_oracle_analysis.py`
(source data: the six `gated_4k_*.jsonl` files; figure written to
`paper/figs/pareto_gated.{pdf,png}`).

**Convention (post-hoc gate, used everywhere).** For input *i* at nominal budget
*b* < 1: gated outcome = `correct_plain(i, b)` if `drop_i ≥ τ = 0.07`, else the
full-KV outcome; gated kept-KV = `n_kept_plain(i,b)/T_i` if the gate opens, else
1.0. Full-KV outcomes come from the *b* = 1.0 rows (Yi-1.5-9B has no *b* = 1.0
rows; its full-KV outcomes are recovered from `correct_gated` of stored-gate-closed
rows, available for 156/200 matrix-suite inputs). This is the same post-hoc
convention the paper already uses for the Qwen2.5-3B SnapKV cell (executed at
τ = 0.04, reported at τ = 0.07).

**Sanity checks.** For the three runs executed at τ = 0.07 (Qwen1.5B, Qwen14B,
Mistral-7B) the reconstruction reproduces the stored `correct_gated` /
`n_kept_gated` columns with **0/8000 mismatches**; for Qwen3B the same holds at
its stored τ = 0.04 (0/3200). Pooled plain/gated accuracies reproduce
`tab:matrix` exactly (0.438/0.559, 0.580/0.839, 0.709/0.853, 0.623/0.810) and
`tab:crossarch` exactly (Yi +0.390, Llama +0.109).

---

## 1. Pareto curves: accuracy vs *achieved* kept-KV fraction

Figure: `paper/figs/pareto_gated.pdf`. Mixed suite (NIAH-MK3 + VT + FWE + QA_1),
x = mean over inputs of kept-KV fraction, y = accuracy. Full data:

| model | b | plain kept | plain acc | gated kept | gated acc | plain acc @ gated kept* | gated adv |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B | 0.0625 | 0.062 | 0.185 | 0.297 | 0.347 | 0.441 | **−0.093** |
| | 0.125 | 0.125 | 0.385 | 0.344 | 0.547 | 0.449 | +0.098 |
| | 0.25 | 0.250 | 0.432 | 0.437 | 0.590 | 0.463 | +0.127 |
| | 0.5 | 0.500 | 0.470 | 0.625 | 0.598 | 0.498 | +0.100 |
| | 0.875 | 0.875 | 0.550 | 0.906 | 0.595 | 0.564 | +0.031 |
| | 1.0 | 1.000 | 0.608 | 1.000 | 0.608 | | |
| Qwen2.5-3B | 0.0625 | 0.062 | 0.190 | 0.541 | 0.610 | 0.661 | **−0.051** |
| | 0.125 | 0.125 | 0.458 | 0.571 | 0.830 | 0.670 | +0.160 |
| | 0.25 | 0.250 | 0.535 | 0.632 | 0.877 | 0.690 | +0.187 |
| | 0.5 | 0.500 | 0.647 | 0.755 | 0.882 | 0.735 | +0.147 |
| | 0.875 | 0.875 | 0.802 | 0.939 | 0.880 | 0.843 | +0.037 |
| | 1.0 | 1.000 | 0.882 | 1.000 | 0.882 | | |
| Qwen2.5-14B | 0.0625 | 0.062 | 0.250 | 0.297 | 0.500 | 0.664 | **−0.164** |
| | 0.125 | 0.125 | 0.560 | 0.344 | 0.810 | 0.704 | +0.106 |
| | 0.25 | 0.250 | 0.625 | 0.437 | 0.875 | 0.775 | +0.100 |
| | 0.5 | 0.500 | 0.820 | 0.625 | 0.920 | 0.860 | +0.060 |
| | 0.875 | 0.875 | 0.930 | 0.906 | 0.935 | 0.932 | +0.003 |
| | 1.0 | 1.000 | 0.940 | 1.000 | 0.940 | | |
| Mistral-7B | 0.0625 | 0.062 | 0.260 | 0.327 | 0.510 | 0.610 | **−0.100** |
| | 0.125 | 0.125 | 0.555 | 0.372 | 0.800 | 0.619 | +0.181 |
| | 0.25 | 0.250 | 0.595 | 0.462 | 0.830 | 0.639 | +0.191 |
| | 0.5 | 0.500 | 0.647 | 0.641 | 0.858 | 0.700 | +0.157 |
| | 0.875 | 0.875 | 0.848 | 0.910 | 0.890 | 0.864 | +0.026 |
| | 1.0 | 1.000 | 0.905 | 1.000 | 0.905 | | |

*linear interpolation of the plain curve at the gated point's achieved kept-KV.
(Budgets 0.375, 0.625, 0.75 omitted for brevity; they interpolate smoothly and
are all gated-positive; the figure and the script print all eight.)

### Verdict on the Pareto claim (state it exactly this way)

The gated curve does **not** dominate everywhere; it dominates on all but the
most aggressive nominal budget, and the honest statement is about the frontier:

1. **Gated points sit at larger kept-KV than their nominal budget** — the
   full-KV fallback raises achieved kept-KV from *b* to roughly
   (1−open) + open·*b*. At matched *achieved* cache size, for every target
   kept-KV fraction **x ≳ 0.34** (Qwen1.5B, Qwen14B), **x ≳ 0.37** (Mistral-7B),
   **x ≳ 0.57** (Qwen3B), the gated curve gives strictly higher accuracy than
   the plain curve — by up to **+12.7 / +18.7 / +10.6 / +19.1 pp** respectively
   (peak advantage, at kept ≈ 0.44 / 0.63 / 0.34 / 0.46). Equivalently: gated is
   the frontier at every nominal budget **b ≥ 0.125**.
2. **At the single most aggressive nominal budget (b = 0.0625) the gated point
   is strictly dominated by plain** in all four models: e.g. Qwen14B gated
   achieves (kept 0.297, acc 0.500) while plain at nominal 0.25 achieves
   (kept 0.250, acc 0.625). Reason: the mixture "6% cache on gate-open inputs +
   100% on the rest" is worse at that cache size than uniform 25–30% eviction —
   near-floor accuracy on the evicted majority is not offset by the full-KV
   minority.
3. **τ = 0.07 puts a floor on achievable compression.** As *b* → 0, gated
   kept-KV → 1 − open-fraction: 0.25 (Qwen1.5B, Qwen14B), 0.28 (Mistral-7B),
   0.51 (Qwen3B). The gate cannot reach cache sizes below that floor; below
   ~0.3 kept-KV the plain evictor is the only (and better) option.

So the correct headline is: *for cache budgets of ≥ ~1/3 of the full KV, gating
moves the accuracy–memory frontier up by 3–19 pp at matched achieved cache size;
below ~1/3 it saturates and plain eviction takes over the frontier.*

---

## 2. Achieved compression at τ = 0.07

Gate-open fraction is a per-input property (independent of *b*). "Compression ×"
= 1 / mean kept-KV. Paper operating points: the `tab:matrix` grid mean (8
budgets 0.0625–0.875; plain grid-mean kept = 0.453 → 2.21×) and the aggressive
end *b* = 0.0625 (plain: 16×).

| model | gate-open frac | kept @ b=0.0625 | kept @ b=0.25 | kept @ b=0.5 | grid-mean kept | compression @ b=0.0625 | grid-mean compression |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B | 0.750 | 0.297 | 0.437 | 0.625 | 0.584 | 3.4× | 1.71× |
| Qwen2.5-3B | 0.490 | 0.541 | 0.632 | 0.755 | 0.728 | 1.8× | 1.37× |
| Qwen2.5-14B | 0.750 | 0.297 | 0.437 | 0.625 | 0.584 | 3.4× | 1.71× |
| Mistral-7B | 0.718 | 0.327 | 0.462 | 0.641 | 0.602 | 3.1× | 1.66× |
| Yi-1.5-9B (4-task) | 0.220 | 0.794 | 0.835 | 0.890 | 0.832¹ | 1.3× | 1.20× |
| Llama-3.1-8B (4-task) | 0.880 | 0.175 | 0.340 | 0.560 | 0.326¹ | 5.7× | 3.07× |

¹ Yi/Llama use the 4-budget grid {0.0625, 0.125, 0.25, 0.5}; their "grid mean"
is over those (plain grid-mean kept = 0.234 → 4.27×), not comparable to the
8-budget rows.

**Bottom line:** at the nominal-16× operating point the deployment actually
delivers **3.1–3.4×** compression on Qwen1.5B/14B/Mistral and only **1.8×** on
Qwen3B (whose gate opens for just 49% of inputs). Averaged over the paper's
budget grid, gating delivers 1.4–1.7× against plain's 2.2×. Yi-1.5-9B is the
known degenerate case (1.2–1.3×, a de-facto full-KV fallback; cf.
`normalized_predictor.md`); Llama-3.1-8B over-opens (0.88) and compresses hard
(5.7×) but transfers poorly on accuracy. Any statement of the matrix Δ should be
accompanied by these kept-KV numbers.

---

## 3. Oracle gap and gate recovery

Oracle = per-input, per-budget max(correct_plain(b), full-KV outcome); pooled
over budgets < 1 on the 4-task suite. Recovery = (gated − plain)/(oracle − plain).

### Per model (pooled)

| model | plain | gated | oracle | Δ gated | Δ oracle | **recovery** |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B | 0.438 | 0.559 | 0.624 | +0.121 | +0.186 | **65.0%** |
| Qwen2.5-3B | 0.580 | 0.839 | 0.886 | +0.259 | +0.306 | **84.8%** |
| Qwen2.5-14B | 0.709 | 0.853 | 0.947 | +0.143 | +0.238 | **60.3%** |
| Mistral-7B | 0.623 | 0.810 | 0.907 | +0.187 | +0.285 | **65.6%** |
| **grand mean (4 models)** | | | | **+0.178** | **+0.254** | **68.9%** (ratio of means 70.0%) |

### Per (model, task) — where the recovered points live

| model | task | Δ gated | Δ oracle | recovery |
|---|---|---:|---:|---:|
| Qwen2.5-1.5B | niah_multikey_3 | +0.484 | +0.489 | 99.0% |
| Qwen2.5-1.5B | vt | +0.000 | +0.136 | 0% |
| Qwen2.5-1.5B | fwe | +0.000 | +0.060 | 0% |
| Qwen2.5-1.5B | qa_1 | +0.000 | +0.059 | 0% |
| Qwen2.5-3B | niah_multikey_3 | +0.734 | +0.734 | 100.0% |
| Qwen2.5-3B | fwe | +0.300 | +0.301 | 99.6% |
| Qwen2.5-3B | vt | +0.004 | +0.140 | 2.7% |
| Qwen2.5-3B | qa_1 | +0.000 | +0.049 | 0% |
| Qwen2.5-14B | niah_multikey_3 | +0.573 | +0.573 | 100.0% |
| Qwen2.5-14B | vt | +0.000 | +0.155 | 0% |
| Qwen2.5-14B | fwe | +0.000 | +0.213 | 0% |
| Qwen2.5-14B | qa_1 | +0.000 | +0.010 | 0% |
| Mistral-7B | niah_multikey_3 | +0.701 | +0.781 | 89.8% |
| Mistral-7B | fwe | +0.046 | +0.219 | 21.1% |
| Mistral-7B | vt | +0.000 | +0.126 | 0% |
| Mistral-7B | qa_1 | +0.000 | +0.012 | 0% |

Interpretation: the gate recovers **essentially all** of the oracle headroom on
the capacity-bound anchor (MK3: 90–100%) — that is what it was calibrated to do —
and **essentially none** of the residual per-input headroom inside the
dilution-prone tasks (vt/fwe/qa: 0–21%), where the oracle's remaining gains come
from scattered inputs whose drop exceeds τ but which would still have been
answered correctly with full KV. The task-level gate is ~69% of a per-input
oracle overall; the missing 31% is *within-task* per-input selection, which the
current scalar D at a single τ does not attempt.

### Llama-family cells (τ = 0.07, post-hoc; 4-task suite, budgets {0.0625–0.5})

| model | plain | gated | oracle | Δ gated | Δ oracle | recovery | gated kept-KV (grid mean) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Yi-1.5-9B | 0.464 | 0.854 | 0.859–0.915² | +0.390 | +0.395 to +0.451² | **86–99%²** | 0.832 |
| Llama-3.1-8B | 0.509 | 0.618 | 0.936 | +0.109 | +0.427 | **25.4%** | 0.326 |

² Yi's full-KV outcome is unknown for the 44/200 always-gate-open inputs; the
range brackets the oracle by assuming those inputs wrong (lower) or correct
(upper) under full KV. The gated reconstruction itself is exact. 6-task pooled
versions: Yi Δ +0.323 (recovery ≤ 98.5%, kept 0.803); Llama Δ +0.089 (recovery
≤ 28.8%, kept 0.410).

Yi's near-perfect "recovery" is an artifact of near-zero compression (kept-KV
0.79–0.89: the gate approximates the full-KV arm of the oracle by simply not
evicting). Llama-3.1-8B is the genuine transfer failure: only a quarter of the
oracle gap is recovered because MK3 half-opens at τ = 0.07. **Context from
`normalized_predictor.md` (z-score predictor):** the raw τ = 0.07 sits at
z = −0.97 in Llama's pooled drop distribution but at z = +0.69 in Yi's — two
opposite miscalibrations. The architecture-normalized gate (z-score with global
θ_z = −0.69, fit on Qwen/Mistral only) lifts Llama to Δ +0.174, i.e. recovery
25.4% → **40.7%** at comparable kept-KV (0.33 → 0.39 grid mean), and converts Yi
into a real compressor (kept 0.83 → 0.36 grid mean) at Δ +0.145 (recovery
32–37% of the bracketed oracle) — trading fallback-inflated accuracy for actual memory
savings.

---

## 4. What to fix in the paper text

1. Never report matrix Δ without kept-KV: add the achieved-compression table
   (snippet in `paper/snippets/pareto_snippet.tex`) and the Pareto figure.
2. Phrase the frontier claim as in §1 above (dominates for kept-KV ≳ 1/3;
   dominated at b = 0.0625; compression floor 1 − open-fraction).
3. Oracle framing: "the τ = 0.07 gate recovers 60–85% (mean 69%) of the
   per-input oracle's headroom on the Qwen/Mistral matrix, almost entirely via
   task-level MK3 protection" is accurate; "gating is near-oracle" is not.
