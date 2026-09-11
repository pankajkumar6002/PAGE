# Headline matrix restricted to b <= 0.25

tau = 0.07. Delta = gated minus plain accuracy, over the same inputs.
The `b < 1.0` column is the convention tab:matrix currently reports;
`b <= 0.25` is the >= 4x regime its own caption directs the reader to.

| cell | Delta (b<1.0) | Delta (b<=0.25) | change | MK3 (b<=0.25) | no-MK3 (b<=0.25) |
|---|---:|---:|---:|---:|---:|
| h2o mistral7b | +0.236 | +0.244 | +0.008 | +0.890 | +0.028 |
| h2o qwen14b | +0.200 | +0.250 | +0.050 | +1.000 | +0.000 |
| h2o qwen15b | +0.162 | +0.163 | +0.001 | +0.650 | +0.000 |
| h2o qwen3b | +0.303 | +0.326 | +0.023 | +0.930 | +0.125 |
| pyramidkv mistral7b | +0.231 | +0.240 | +0.009 | +0.890 | +0.023 |
| pyramidkv qwen14b | +0.228 | +0.250 | +0.022 | +1.000 | +0.000 |
| pyramidkv qwen15b | +0.152 | +0.160 | +0.008 | +0.640 | +0.000 |
| pyramidkv qwen3b | +0.329 | +0.365 | +0.036 | +0.930 | +0.177 |
| snapkv mistral7b | +0.187 | +0.243 | +0.056 | +0.890 | +0.028 |
| snapkv qwen14b | +0.143 | +0.250 | +0.107 | +1.000 | +0.000 |
| snapkv qwen15b | +0.121 | +0.161 | +0.040 | +0.643 | +0.000 |
| snapkv qwen3b | +0.259 | +0.378 | +0.119 | +0.930 | +0.194 |
| streamingllm mistral7b | +0.263 | +0.263 | +0.000 | +0.890 | +0.053 |
| streamingllm qwen14b | +0.250 | +0.250 | +0.000 | +1.000 | +0.000 |
| streamingllm qwen15b | +0.163 | +0.163 | +0.000 | +0.650 | +0.000 |
| streamingllm qwen3b | +0.430 | +0.430 | +0.000 | +0.930 | +0.263 |
| **grand mean** | **+0.2286** | **+0.2584** | **+0.0299** | | **+0.0558** |

Cells: 16. Restricting to the aggressive regime moves the grand mean +0.2286 -> +0.2584 (+0.0299).

## Matched-grid control (the honest comparison)

The cells do not share a budget grid: SnapKV carries 8 budgets, the
other three policies only {0.125, 0.25, 0.5}. The raw shift above
therefore mixes the restriction with the grid. Restricted to the three
budgets every cell has:

| grid | grand mean |
|---|---:|
| {0.125, 0.25, 0.5} | +0.2402 |
| {0.125, 0.25} (aggressive) | +0.2569 |
| **restriction effect** | **+0.0167** |

The direction survives the control: restricting to the aggressive
budgets raises the mean on a like-for-like grid too. Quote the
matched-grid effect (+0.0167), not the raw one.

## What this settles

Restricting the headline to the aggressive (>= 4x) budget regime does
not deflate it, as averaging over the disowned moderate-budget regime
might be suspected to do. The restricted mean is HIGHER, so the
reported +22.9pp is if anything conservative with respect to the
budgets the paper's own caption endorses.

This does not by itself answer the baseline objection: both columns
use the shared single keep-mask. The per-head Ada-KV analysis
(`adakv_matrix.md`) replaces the plain arm with per-head allocation
and is the experiment that settles that question.

## Which 'clean per-head SnapKV' figure is meant

Two different per-head allocations of the same SnapKV scores appear
elsewhere in the paper: **0.32 is the Ada-KV per-head row**, and
**0.20 is the uniform per-head row**. Any citation of "clean per-head
SnapKV" should name which one it means.

## Checks

- [PASS] grand mean b<1.0 == 0.2286 (got 0.2286)
- [PASS] grand mean b<=0.25 == 0.2584 (got 0.2584)
- [PASS] matched-grid b in {0.125,0.25,0.5} == 0.2402 (got 0.2402)
- [PASS] matched-grid b in {0.125,0.25} == 0.2569 (got 0.2569)
- [PASS] restriction raises the mean on a matched grid (got +0.0167)
- [PASS] every restricted cell >= its unrestricted value (got ok)
