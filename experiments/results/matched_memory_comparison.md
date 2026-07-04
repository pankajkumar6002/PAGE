# Matched-MEMORY comparison: does PAGE beat a strong evictor at a fixed *achieved* kept-KV fraction?

Reproduce:
- Part 1 (analysis-only): `.venv/bin/python experiments/scripts/matched_memory_analysis.py`
  (source: the six `gated_4k_*.jsonl` SnapKV files + the H2O / StreamingLLM /
  PyramidKV variant files).
- Part 2 PAGE-SnapKV side (analysis-only):
  `.venv/bin/python experiments/scripts/matched_memory_page_snapkv_qwen3.py`
  (source: `gated_4k_qwen3_4b_merged.jsonl`).
- Part 2 plain-DBTrimKV side (native run):
  `external/trimkv/venv/bin/python external/trimkv/run_plain_dbtrimkv_30pct.py`
  (reuses `run_gated_dbtrimkv.py` helpers, plain policy only, at
  `memory_size ∈ {1024,1152,1280}` → 26/29/32 % kept-KV); result file
  `experiments/results/plain_dbtrimkv_qwen3_4k_30pct.jsonl`. Aggregate with
  `experiments/scripts/matched_memory_dbtrimkv.py`.

**This answers the reviewer's strongest objection.** The paper compares
gated-SnapKV (PAGE) vs plain SnapKV at matched *nominal* budget, but the gate's
full-KV fallback means the gated policy always *holds more memory*. The honest
question is: at a fixed **achieved kept-KV fraction**, does PAGE beat a strong
evictor? Below we answer at matched achieved cache and report the crossover
kept-KV fraction, per model.

**Convention.** Post-hoc gate at τ = 0.07 (identical to
`pareto_oracle_analysis.md`): for input *i* at nominal budget *b* < 1,
`gated_correct = correct_plain(i,b)` if `drop_i ≥ τ` else `full_correct(i)`;
`gated_kept = n_kept_plain/T` if the gate opens else `1.0`. PAGE's **achieved**
mean kept-KV therefore floors at `1 − p_open` as *b* → 0 (the full-KV fallback).
Post-hoc reconstruction reproduces the stored `correct_gated`/`n_kept_gated`
columns with 0 mismatches for every run executed at τ = 0.07 (Qwen1.5B, Qwen14B,
Mistral-7B; Qwen3B stored at τ = 0.04, analysed at τ = 0.07 as in the paper).

---

## (i) PAGE vs plain at MATCHED achieved cache — table + crossover per model

For each model × base-evictor we build two accuracy-vs-*achieved*-kept-KV
curves — plain eviction sweeping *b*, and PAGE sweeping *b* — then linearly
interpolate **both** at fixed achieved-kept targets and report **PAGE − plain**
at matched cache. `n/r` = PAGE cannot reach that cache size (target is below its
full-KV-fallback floor `1 − p_open`); plain is the only option there. "Crossover
kept-KV" is the achieved-cache fraction below which plain eviction is on/above
the frontier (plain ≥ PAGE) and above which PAGE improves it.

| model | base-evictor | gate-open p | PAGE floor kept | ΔAcc @0.25 | ΔAcc @0.35 | ΔAcc @0.50 | **crossover kept** | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Qwen2.5-1.5B | SnapKV       | 0.75 | 0.297 | n/r | **+0.100** | **+0.127** | **0.320** | plain wins < 0.32, PAGE above |
| Qwen2.5-1.5B | H2O          | 0.75 | 0.344 | n/r | +0.003 | **+0.105** | **0.348** | plain wins < 0.35, PAGE above |
| Qwen2.5-1.5B | PyramidKV    | 0.75 | 0.344 | n/r | **+0.112** | **+0.132** | **0.344** | PAGE wins wherever it can operate (≥ floor) |
| Qwen2.5-1.5B | StreamingLLM | 0.75 | 0.258 | n/r | +0.016 | +0.013 | — | *degenerate sweep* (see note) |
| Mistral-7B   | SnapKV       | 0.72 | 0.327 | n/r | +0.042 | **+0.189** | **0.343** | plain wins < 0.34, PAGE above |
| Mistral-7B   | H2O          | 0.72 | 0.372 | n/r | n/r | **+0.112** | **0.372** | PAGE wins wherever it can operate (≥ floor) |
| Mistral-7B   | PyramidKV    | 0.72 | 0.372 | n/r | n/r | **+0.191** | **0.372** | PAGE wins wherever it can operate (≥ floor) |
| Mistral-7B   | StreamingLLM | 0.72 | 0.289 | n/r | +0.033 | +0.025 | — | *degenerate sweep* (see note) |
| Qwen2.5-3B   | SnapKV       | 0.49 | 0.541 | n/r | n/r | n/r | **0.548** | plain wins < 0.55, PAGE above |
| Qwen2.5-14B  | SnapKV       | 0.75 | 0.297 | n/r | **+0.105** | **+0.088** | **0.325** | plain wins < 0.33, PAGE above |

*Degenerate note.* StreamingLLM's achieved kept-KV does **not** vary with the
nominal budget in these runs (fixed sink + recent window: kept ≈ 0.01 at every
*b*, identical accuracy across budgets), so the matched-cache sweep has only two
distinct achieved-cache points and the crossover is ill-posed. We report it for
completeness but draw no conclusion from it.

**Crossover kept-KV fractions (the headline of Part i):**

- Qwen2.5-1.5B: **≈ 0.32** (SnapKV), **≈ 0.35** (H2O), **≈ 0.34** (PyramidKV floor)
- Mistral-7B: **≈ 0.34** (SnapKV), **≈ 0.37** (H2O/PyramidKV floor)
- Qwen2.5-14B: **≈ 0.33** (SnapKV)
- Qwen2.5-3B: **≈ 0.55** (SnapKV) — high because the gate opens for only 49 % of
  inputs, so PAGE's cache floors at 0.54 and it can barely compress.

**Reading of the table.** Above the crossover (kept-KV ≳ 0.30–0.35 for the
1.5B/7B/14B models; ≳ 0.55 for the 3B), PAGE strictly improves the
accuracy–memory frontier — by **+9 to +19 pp** at 0.50 kept-KV. Below the
crossover, plain eviction is on or above the frontier, and moreover PAGE
*cannot physically reach* those cache sizes: its full-KV fallback floors achieved
kept-KV at `1 − p_open` (0.25–0.54 depending on model). This rigorously confirms
and sharpens the Pareto finding: **PAGE is a moderate-compression method
(≤ ~3× compression); it does not compete in the aggressive regime.**

---

## (ii) DBTrimKV at matched memory — head-to-head at ~30 % achieved kept-KV

We run a natively-executed, trained evictor (DBTrimKV on its own
`ngocbh/DBTrimKV-Qwen3-4B-Instruct-2507` checkpoint) at a `memory_size` that
yields ~30 % kept-KV, and compare against PAGE-SnapKV on Qwen3-4B at the same
**achieved** 30 % memory (not the 1.7 %-gate-fires nominal setting). Both on the
same RULER-4K 4-task suite (niah_multikey_3, vt, fwe, qa_1), first 30 examples
per task (120 inputs); DBTrimKV per-head kept-KV fraction = `memory_size / T`,
mean T ≈ 3742.

**PAGE-SnapKV on Qwen3-4B (analysis of `gated_4k_qwen3_4b_merged`, matched 120-input suite; A_full = 0.858):**

| operating point | τ | b | achieved kept-KV | accuracy |
|---|---:|---:|---:|---:|
| **paper operating point** | 0.07 | any (0.0625–0.5) | **0.98–0.99** | 0.858 |
| forced to ~30 %, aggressive b | 0.0225 | 0.0625 | 0.305 | **0.425** |
| forced to ~30 %, milder b | 0.020 | 0.125 | 0.329 | **0.592** |
| plain SnapKV reference @ 30 % kept | — | interp | 0.300 | 0.565 |

At its **designed τ = 0.07 the gate fires for only 1.7 %** of inputs, so PAGE
holds ~98 % of the KV cache — **it has no 30 %-memory operating point at all.**
Forcing 30 % memory requires collapsing τ to ≈ 0.02 (gate opens ~74 %), which
drops accuracy to **0.42–0.59** — no better than *plain* SnapKV (0.565) at the
same cache, because "6 % cache on the 74 % opened inputs + 100 % on the rest" is
a dominated mixture at aggressive cache (the same effect as Part i).

**Plain DBTrimKV on Qwen3-4B (native run, this work):**

| memory_size | kept-KV | accuracy | source |
|---:|---:|---:|---|
| 128 | 0.039 | 0.475 | prior run |
| 256 | 0.073 | 0.617 | prior run |
| 512 | 0.142 | 0.758 | prior run |
| 1024 | 0.281 | 0.775 | this run |
| **1152** | **0.315** | **0.775** | this run |
| 1280 | 0.350 | 0.783 | this run |

Interpolated plain-DBTrimKV accuracy @ 30 % kept-KV = **0.775**. Note DBTrimKV
plateaus at ~0.78 above ~28 % kept: its remaining error is on `fwe`
(frequent-word extraction, stuck at 0.33) and `qa_1` (~0.78); `niah_multikey_3`
and `vt` are already at 1.00.

**Head-to-head at matched ~30 % achieved memory:**

| method @ ~30 % kept-KV | achieved kept | accuracy |
|---|---:|---:|
| **plain DBTrimKV** (M = 1152) | 0.315 | **0.775** |
| PAGE-SnapKV (best ~30 %, τ = 0.02, b = 0.125) | 0.329 | 0.592 |
| PAGE-SnapKV (closest 0.30, τ = 0.0225, b = 0.0625) | 0.305 | 0.425 |
| plain SnapKV (reference) | 0.30 | 0.565 |
| PAGE-SnapKV @ paper τ = 0.07 (NOT 30 %) | **0.98** | 0.858 |

**Verdict (ii).** At matched ~30 % memory, **plain DBTrimKV (0.775) decisively
beats PAGE-SnapKV (0.42–0.59)** on Qwen3-4B — a **+0.18 to +0.35** gap in
DBTrimKV's favour, and DBTrimKV even beats plain SnapKV (0.565) there.
PAGE-SnapKV's only competitive number (0.858) is obtained by **holding ~98 % of
the cache**, i.e. by spending ~3× the memory of the 30 % DBTrimKV point. The
DBTrimKV report's headline +0.233 (0.617 → 0.850) gated win is likewise a
**memory-spending win** (gate closed → full paged KV, ≈ 14.5× the cache of plain
DBTrimKV at the same `memory_size`), **not** a frontier win. **If PAGE only wins
by spending more memory — on Qwen3-4B, it does.**

---

## (iii) Honest verdict — does PAGE improve the accuracy–memory frontier at matched memory?

**Yes, but only in the moderate-compression regime, and only on models whose
gate opens often.**

1. **Where PAGE helps.** For kept-KV **above the crossover** (≈ 0.30–0.35 on
   Qwen2.5-1.5B, Mistral-7B, Qwen2.5-14B across SnapKV/H2O/PyramidKV), PAGE
   improves the frontier at matched achieved cache by **+3 to +19 pp**. This is a
   real, matched-memory win — the gate spends its extra cache on exactly the
   inputs that need it.

2. **Where plain wins.** For kept-KV **below the crossover** (aggressive regime,
   ≥ ~3× compression), plain eviction is on or above the frontier, and PAGE
   *cannot even reach* those cache sizes: its full-KV fallback floors achieved
   kept-KV at `1 − p_open` (0.25–0.54). The paper's most aggressive nominal
   operating point (b = 0.0625, "16×") actually delivers only 1.8–3.4× and sits
   *below* the frontier — dominated by plain eviction at matched cache.

3. **Where PAGE is useless.** On models where the gate rarely fires
   (Qwen2.5-3B: p_open = 0.49, crossover 0.55; **Qwen3-4B: p_open = 0.017**),
   PAGE provides essentially **no compression** — on Qwen3-4B it holds ~98 % of
   the cache at τ = 0.07 — and at matched 30 % memory it is beaten by both a
   trained evictor (DBTrimKV) and even by plain SnapKV.

**Crossover kept-KV fractions (deliverable):** ≈ **0.32** (Qwen2.5-1.5B/SnapKV),
**0.35** (Qwen2.5-1.5B/H2O), **0.34** (Qwen2.5-1.5B/PyramidKV floor), **0.34**
(Mistral-7B/SnapKV), **0.37** (Mistral-7B/H2O & PyramidKV floor), **0.33**
(Qwen2.5-14B/SnapKV), **0.55** (Qwen2.5-3B/SnapKV).

**One-line honest headline.** *At matched achieved memory, PAGE improves the
accuracy–memory frontier only for cache budgets ≳ 1/3 of the full KV (≤ ~3×
compression) and only when its gate opens often; below ~1/3 kept-KV, or on
low-gate-open models such as Qwen3-4B, PAGE cannot compress and a strong evictor
(plain SnapKV, DBTrimKV) owns the frontier — PAGE's apparent wins there come
from spending more memory, not from a better frontier.*
