# 95% confidence intervals for the paper's headline numbers

Computed 2026-07-02 from the raw per-input jsonls in
`experiments/results/` with `.venv/bin/python`
(script: session scratchpad `ci_analysis.py`; seed 20260702).

**Methods.**

- **Gating deltas (matrix, cross-arch, scaling, DBTrimKV): cluster bootstrap over
  inputs.** Unit of resampling = the (task, id) input; each input contributes its
  per-budget paired (gated − plain) differences, averaged into one per-input
  delta; 10,000 bootstrap replicates; 95% percentile interval. This respects the
  pairing (gated and plain share the same input at every budget) and the
  within-input correlation across budgets.
- **Gated outcome reconstructed post-hoc at tau = 0.07 for every cell:**
  gated(b) = correct_plain(b) if the input's drop >= 0.07, else the full-KV
  outcome (correct_plain at budget 1.0 for the same (task, id)). This is the
  paper's stated convention; it is applied uniformly because one run
  (`gated_4k_qwen3b.jsonl`) was executed at tau = 0.04, so its stored
  `correct_gated` is never used. Every recomputed cell mean reproduces the
  paper's printed plain/gated/Delta values exactly (Qwen2.5-3B SnapKV gives
  0.839 / +0.259 as the paper reports post-hoc). Two files have no budget-1.0
  record: Yi-1.5-9B (executed at tau = 0.07; the stored `correct_gated` of
  gate-closed inputs is the full-KV generation and is used as such) and
  DBTrimKV (same situation, budgets are absolute {128, 256, 512}).
- **Partition recovery rates rho: Wilson 95% score interval** on the binomial
  count (n_rec, N). Counts are the corrected values (audit of 2026-07-02) plus
  the two completed Mistral sweeps `ruler_4k_mistral7b_qa2_mv.jsonl` /
  `ruler_16k_mistral7b_qa2_mv.jsonl`, whose rho was recomputed from pred/gold
  with the `reanalyze_ruler.py` convention (vt/fwe/cwe/niah_multivalue = ALL
  golds substring-in-pred lowercased; others ANY; rho = full-KV wrong AND some
  b < 1.0 correct). Recomputation reproduces `ruler_mistral7b_qa2_mv_final.md`
  exactly (qa_2: 1/100, 0/100; niah_multivalue: 6/100, 21/100).

Budget grids: SnapKV 4K/16K cells average over 8 budgets
{0.0625...0.875}; H2O/StreamingLLM/PyramidKV files contain {0.125, 0.25, 0.5}
(+1.0); Yi/Llama use the 4-budget grid {0.0625, 0.125, 0.25, 0.5}; DBTrimKV
uses absolute cache sizes {128, 256, 512}. Cross-arch rows use the same 4-task
mixed suite (NIAH-MK3 + VT + FWE + QA_1) as the matrix, N = 50 per task.

---

## 1. Headline 4x4 matrix (Table `tab:matrix`), Delta = gated − plain

Cluster bootstrap, 10,000 reps, percentile CI. n = number of (task, id) inputs.

| Model | Evictor | plain | gated | Delta | 95% CI | n |
|---|---|---:|---:|---:|---|---:|
| Qwen2.5-1.5B | SnapKV       | 0.438 | 0.559 | +0.121 | [+0.093, +0.151] | 400 |
| Qwen2.5-1.5B | H2O          | 0.196 | 0.357 | +0.162 | [+0.126, +0.199] | 400 |
| Qwen2.5-1.5B | StreamingLLM | 0.033 | 0.195 | +0.163 | [+0.128, +0.200] | 400 |
| Qwen2.5-1.5B | PyramidKV    | 0.417 | 0.570 | +0.152 | [+0.118, +0.188] | 400 |
| Qwen2.5-3B   | SnapKV       | 0.580 | 0.839 | +0.259 | [+0.225, +0.295] | 400 |
| Qwen2.5-3B   | H2O          | 0.339 | 0.642 | +0.303 | [+0.261, +0.347] | 400 |
| Qwen2.5-3B   | StreamingLLM | 0.052 | 0.482 | +0.430 | [+0.383, +0.477] | 400 |
| Qwen2.5-3B   | PyramidKV    | 0.536 | 0.865 | +0.329 | [+0.286, +0.372] | 400 |
| Qwen2.5-14B  | SnapKV       | 0.709 | 0.853 | +0.143 | [+0.108, +0.181] | 200 |
| Qwen2.5-14B  | H2O          | 0.668 | 0.868 | +0.200 | [+0.152, +0.250] | 200 |
| Qwen2.5-14B  | StreamingLLM | 0.080 | 0.330 | +0.250 | [+0.190, +0.310] | 200 |
| Qwen2.5-14B  | PyramidKV    | 0.632 | 0.860 | +0.228 | [+0.175, +0.285] | 200 |
| Mistral-7B   | SnapKV       | 0.623 | 0.810 | +0.187 | [+0.154, +0.220] | 400 |
| Mistral-7B   | H2O          | 0.400 | 0.636 | +0.236 | [+0.196, +0.278] | 400 |
| Mistral-7B   | StreamingLLM | 0.102 | 0.365 | +0.263 | [+0.220, +0.305] | 400 |
| Mistral-7B   | PyramidKV    | 0.593 | 0.823 | +0.231 | [+0.191, +0.272] | 400 |

All 16 point estimates reproduce the paper exactly (Mistral PyramidKV plain is
0.5925, printed 0.593 in the paper, 0.592 under round-half-even; Delta
identical). **Every one of the 16 CIs excludes 0**; the smallest lower bound is
+0.093 (Qwen1.5B SnapKV). Grand mean +0.229 as the paper states.

## 2. Cross-architecture SnapKV (Table `tab:crossarch`), 4-budget grid, mixed suite

| Model | plain | gated | Delta | 95% CI | n |
|---|---:|---:|---:|---|---:|
| Yi-1.5-9B    | 0.464 | 0.854 | +0.390 | [+0.338, +0.444] | 200 |
| Llama-3.1-8B | 0.509 | 0.618 | +0.109 | [+0.069, +0.154] | 200 |

Both exclude 0. (For reference, over all six tasks in those files rather than
the 4-task mixed suite: Yi +0.323 [+0.282, +0.364], Llama +0.089
[+0.060, +0.120] — also both positive.)

## 3. 8-cell scaling table (Table `tab:scaling`), Delta at tau = 0.07

| Model | Context | plain | gated | Delta | 95% CI | n | source |
|---|---|---:|---:|---:|---|---:|---|
| Qwen2.5-1.5B | 4K  | 0.438 | 0.559 | +0.121 | [+0.093, +0.151] | 400 | gated_4k_qwen15b.jsonl |
| Qwen2.5-3B   | 4K  | 0.580 | 0.839 | +0.259 | [+0.225, +0.295] | 400 | gated_4k_qwen3b.jsonl |
| Qwen2.5-14B  | 4K  | 0.709 | 0.853 | +0.143 | [+0.108, +0.181] | 200 | gated_4k_qwen14b.jsonl |
| Mistral-7B   | 4K  | 0.623 | 0.810 | +0.187 | [+0.154, +0.220] | 400 | gated_4k_mistral7b.jsonl |
| Qwen2.5-1.5B | 16K | 0.372 | 0.404 | +0.032 | [+0.013, +0.053] | 200 | gated_16k_qwen15b_sdpa.jsonl |
| Qwen2.5-3B   | 16K | 0.530 | 0.564 | **+0.034** | **[−0.002, +0.071]** | 200 | gated_16k_qwen3b.jsonl |
| Qwen2.5-14B  | 16K | 0.796 | 0.867 | +0.071 | [+0.034, +0.107] | 120 | gated_16k_qwen14b.jsonl |
| Mistral-7B   | 16K | 0.535 | 0.651 | +0.116 | [+0.079, +0.156] | 200 | gated_16k_mistral7b.jsonl |

**Qwen2.5-3B 16K is the single headline delta whose CI includes 0**
(+0.034, [−0.002, +0.071]). The paper already flags this cell as the
gate-misfire case; the CI makes that quantitative. All seven other cells
exclude 0. The paper's Qwen-14B 16K row is the 4-task mixed suite
(`gated_16k_qwen14b.jsonl` only, n = 120, matches the printed
0.796/0.867/+0.071); pooling in the qa_2 + niah_multivalue inputs of
`gated_16k_qwen14b_extra.jsonl` gives a 6-task variant +0.047
[+0.018, +0.076] (n = 180), still positive.

## 4. Headline showcases

| Quantity | point | 95% CI | method |
|---|---:|---|---|
| Mistral 4K NIAH-MK3 gated acc at b = 0.0625 | 0.89 (89/100) | [0.814, 0.937] | Wilson |
| ...same at b in {0.125, 0.25} | 0.89 (89/100) | [0.814, 0.937] | Wilson |
| ...plain at b = 0.0625 | 0.00 (0/100) | [0.000, 0.037] | Wilson |
| Delta at b = 0.0625 | +0.89 | — (gated CI [0.814, 0.937] vs plain [0.000, 0.037], disjoint) | |
| DBTrimKV wrapper Delta (RULER 4K, Qwen3-4B-Instruct) | +0.233 (plain 0.617, gated 0.850) | [+0.172, +0.297] | cluster bootstrap, n = 120 inputs x 3 budgets |

The showcase reconstruction confirms 10/100 gate-open false positives and
full-KV accuracy 99/100, as stated in the paper.

## 5. Partition table (Table `tab:partition`): rho with Wilson 95% intervals

Bold in the paper = rho >= 0.05. "Call" = whether the Wilson CI settles the
threshold question: **clear-bold** (lower bound >= 0.05), **clear-non-bold**
(upper bound <= 0.05), else **ambiguous** (CI straddles 0.05).

| Task | Cell | rho (n/N) | Wilson 95% | paper call | under Wilson |
|---|---|---:|---|---|---|
| VT | Qwen1.5B 4K | 0.06 (6/100) | [0.028, 0.125] | bold | ambiguous |
| VT | Qwen1.5B 16K | 0.19 (19/100) | [0.125, 0.278] | bold | clear |
| VT | Qwen3B 4K | 0.00 (0/100) | [0.000, 0.037] | non-bold | clear |
| VT | Qwen3B 16K | 0.05 (5/100) | [0.022, 0.112] | bold | ambiguous |
| VT | Qwen14B 4K | 0.00 (0/50) | [0.000, 0.071] | non-bold | ambiguous |
| VT | Qwen14B 16K | 0.03 (1/30) | [0.006, 0.167] | non-bold | ambiguous |
| VT | Mistral 4K | 0.00 (0/100) | [0.000, 0.037] | non-bold | clear |
| VT | Mistral 16K | 0.32 (16/50) | [0.208, 0.458] | bold | clear |
| FWE | Qwen1.5B 4K | 0.06 (6/100) | [0.028, 0.125] | bold | ambiguous |
| FWE | Qwen1.5B 16K | 0.04 (4/100) | [0.016, 0.098] | non-bold | ambiguous |
| FWE | Qwen3B 4K | 0.01 (1/100) | [0.002, 0.054] | non-bold | ambiguous (barely: upper 0.054) |
| FWE | Qwen3B 16K | 0.32 (32/100) | [0.237, 0.417] | bold | clear |
| FWE | Qwen14B 4K | 0.00 (0/50) | [0.000, 0.071] | non-bold | ambiguous |
| FWE | Qwen14B 16K | 0.20 (6/30) | [0.095, 0.373] | bold | clear |
| FWE | Mistral 4K | 0.02 (2/100) | [0.006, 0.070] | non-bold | ambiguous |
| FWE | Mistral 16K | 0.22 (11/50) | [0.128, 0.352] | bold | clear |
| QA_1 | Qwen1.5B 4K | 0.08 (8/100) | [0.041, 0.150] | bold | ambiguous |
| QA_1 | Qwen1.5B 16K | 0.14 (14/100) | [0.085, 0.221] | bold | clear |
| QA_1 | Qwen3B 4K | 0.03 (3/100) | [0.010, 0.085] | non-bold | ambiguous |
| QA_1 | Qwen3B 16K | 0.08 (8/100) | [0.041, 0.150] | bold | ambiguous |
| QA_1 | Qwen14B 4K | 0.06 (3/50) | [0.021, 0.162] | bold | ambiguous |
| QA_1 | Qwen14B 16K | 0.10 (3/30) | [0.035, 0.256] | bold | ambiguous |
| QA_1 | Mistral 4K | 0.04 (4/100) | [0.016, 0.098] | non-bold | ambiguous |
| QA_1 | Mistral 16K | 0.00 (0/50) | [0.000, 0.071] | non-bold | ambiguous |
| QA_2 | Qwen1.5B 4K | 0.08 (8/100) | [0.041, 0.150] | bold | ambiguous |
| QA_2 | Qwen1.5B 16K | 0.05 (5/100) | [0.022, 0.112] | bold | ambiguous |
| QA_2 | Qwen3B 4K | 0.04 (4/100) | [0.016, 0.098] | non-bold | ambiguous |
| QA_2 | Qwen3B 16K | -- | -- | not run | -- |
| QA_2 | Qwen14B 4K | 0.02 (1/50) | [0.004, 0.105] | non-bold | ambiguous |
| QA_2 | Qwen14B 16K | 0.03 (1/30) | [0.006, 0.167] | non-bold | ambiguous |
| QA_2 | Mistral 4K | 0.01 (1/100) | [0.002, 0.054] | non-bold | ambiguous (barely: upper 0.054) |
| QA_2 | Mistral 16K | 0.00 (0/100) | [0.000, 0.037] | non-bold | clear |
| niah_multivalue | Qwen1.5B 4K | 0.07 (7/100) | [0.034, 0.137] | bold | ambiguous |
| niah_multivalue | Qwen1.5B 16K | 0.10 (10/100) | [0.055, 0.174] | bold | clear |
| niah_multivalue | Qwen3B 4K | 0.02 (2/100) | [0.006, 0.070] | non-bold | ambiguous |
| niah_multivalue | Qwen3B 16K | 0.13 (13/100) | [0.078, 0.210] | bold | clear |
| niah_multivalue | Qwen14B 4K | 0.08 (4/50) | [0.032, 0.188] | bold | ambiguous |
| niah_multivalue | Qwen14B 16K | 0.07 (2/30) | [0.018, 0.213] | bold | ambiguous |
| niah_multivalue | Mistral 4K | 0.06 (6/100) | [0.028, 0.125] | bold | ambiguous |
| niah_multivalue | Mistral 16K | 0.21 (21/100) | [0.142, 0.300] | bold | clear |
| NIAH-MK3 | Qwen1.5B 4K | 0.01 (1/100) | [0.002, 0.054] | non-bold | ambiguous (barely) |
| NIAH-MK3 | Qwen1.5B 16K | 0.00 (0/46) | [0.000, 0.077] | non-bold | ambiguous |
| NIAH-MK3 | Qwen3B 4K | 0.00 (0/100) | [0.000, 0.037] | non-bold | clear |
| NIAH-MK3 | Qwen3B 16K | 0.02 (1/50) | [0.004, 0.105] | non-bold | ambiguous |
| NIAH-MK3 | Qwen14B 4K | 0.00 (0/50) | [0.000, 0.071] | non-bold | ambiguous |
| NIAH-MK3 | Qwen14B 16K | 0.00 (0/30) | [0.000, 0.114] | non-bold | ambiguous |
| NIAH-MK3 | Mistral 4K | 0.00 (0/100) | [0.000, 0.037] | non-bold | clear |
| NIAH-MK3 | Mistral 16K | 0.00 (0/50) | [0.000, 0.071] | non-bold | ambiguous |

## Which conclusions survive?

**All causal/headline deltas survive.** Every one of the 16 matrix cells has a
95% bootstrap CI strictly above 0 (smallest lower bound +9.3pp), so the paper's
central claim — the wrapper is positive for every base evictor on every model
with a single tau — holds with interval-level support, as do both cross-arch
cells (Yi lower bound +33.8pp, Llama +6.9pp), the Mistral NIAH-MK3 showcase
(gated Wilson interval [0.814, 0.937] disjoint from plain [0.000, 0.037] at
b = 0.0625), and the DBTrimKV head-to-head (+23.3pp, [+17.2, +29.7]pp). The
one headline delta that is *not* individually significant is the scaling
table's **Qwen2.5-3B 16K cell (+0.034, CI [−0.002, +0.071])** — exactly the
cell the paper already narrates as the gate-misfire regime, so the CI supports
rather than undermines the prose, but the paper should not describe that cell
as a positive result on its own.

**The partition is directional at the per-cell level.** With N = 30–100 per
cell, a Wilson interval around the rho >= 0.05 threshold is wide: only 9 of
the 22 bold cells clear the threshold outright (all six 16K bold cells for
VT/FWE/niah_multivalue on Qwen1.5B/3B/14B/Mistral, plus QA_1 Qwen1.5B 16K,
niah_multivalue Qwen1.5B 16K and Mistral 16K), and only 5 non-bold cells
(the 0/100 cells, plus QA_2 Mistral 16K) exclude it; the remaining 33
data-bearing cells straddle 0.05. No bold cell has an interval *below* 0.05
and no non-bold cell has an interval *above* it, so no call is contradicted —
merely under-powered. The robust versions of the partition claims are the
aggregate ones: NIAH-MK3's pooled recovery across all eight cells is 2/526
(rho = 0.004, Wilson [0.001, 0.014], upper bound far below 0.05), versus e.g.
pooled niah_multivalue 65/680 (0.096, [0.076, 0.120]) — the task-family
separation is unambiguous even though single small-N cells (especially the
N = 30 Qwen-14B 16K column, where 0/30 still allows rho up to 0.114) are not.
Individual borderline bold calls that most need the "directional" caveat:
VT Qwen3B 16K (5/100), QA_2 Qwen1.5B 16K (5/100), niah_multivalue Qwen14B 16K
(2/30), and QA_1 Qwen14B 4K (3/50).

**Notes.** (i) The Qwen2.5-3B SnapKV matrix cell was verified by exact tau = 0.07
post-hoc reconstruction of the tau = 0.04 run: 0.839 / +0.259, matching the
paper. (ii) All other gated cells' stored outcomes agree with the post-hoc
reconstruction. (iii) The two new Mistral sweeps reproduce the values now
printed in the paper's partition table (qa_2: 0.01 / 0.00;
niah_multivalue: 0.06 / 0.21).
