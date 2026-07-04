# Protection-matched ablation (Garcia et al., "Protection Is All You Need")

**Question.** Garcia et al. claim structural protection (protected prefix + protected
suffix, 10% each) dominates scorer choice. Our paper protects only n_sink=4 prefix
tokens + obs_window=32 recency tokens. Does our task partition — eviction helps
VT / FWE / QA_1, catastrophically hurts NIAH-MultiKey-3 — survive under
Garcia-scale protection?

**Verdict: the partition is NOT an artifact of light protection.** NIAH-MK3 still
collapses (0.65 → 0.09 at the protection floor, an 86% relative loss), the
dilution-side gains survive (FWE floor accuracy 0.31 > 0.22 full-KV; ρ_FWE doubles
to 0.13; ρ_QA1 = 0.07), and heavy protection actively *hurts* the scorer-dependent
tasks at both ends (MK3 at high budgets, VT at low budgets).

## Run config

| | |
|---|---|
| script | `experiments/scripts/ruler_sweep.py` |
| model | Qwen/Qwen2.5-1.5B-Instruct (bf16, eager attention, one-pass scoring) |
| data | RULER 4096 (`simonjegou/ruler`), tasks niah_multikey_3, vt, fwe, qa_1, 100 examples each |
| budgets | 1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.1875, 0.125, 0.0625 |
| protection | `--n_sink 410 --obs_window 410` (4096 × 10% ≈ 410, i.e. 10% prefix + 10% suffix) |
| eviction | snapkv; seed 20260603 (default); GPU 1 (A100 80GB, ~4.4 GB used, no OOM, no `--two_pass` needed) |
| output | `experiments/results/ruler_4k_qwen15b_garcia_protect.jsonl` (4000 records); log `experiments/logs/ruler_4k_qwen15b_garcia_protect.log`; wall time 1978 s |
| baselines | `ruler_mk3_4k_snapkv.jsonl`, `ruler_vt_fwe_4k_snapkv.jsonl`, `ruler_qa_4k_qwen.jsonl` (n_sink=4, obs_window=32, same model/config/N) |
| analysis | `experiments/scripts/analyze_garcia_protection.py`; correctness per `reanalyze_ruler.py` conventions (vt/fwe/cwe/niah_multivalue: ALL golds substring in lowercased pred; others: ANY) |

## Confound note (must accompany any use of these numbers)

In `ruler_sweep.py`, `obs_window` serves double duty: it is BOTH the SnapKV
scoring window (attention from the last `obs_window` query rows is averaged to
score middle tokens, `prefill_and_score`) AND the recency-protection span
(`derive_keep_mask` unconditionally keeps the last `obs_window` positions).
Setting `obs_window=410` therefore changes the scorer as well as the protection:
importance scores are averaged over 410 query rows instead of 32, diluting the
selectivity of the SnapKV signal. This run is thus a "protection-heavy SnapKV
variant" rather than a pure protection ablation — which is the spirit of Garcia's
protocol (their protected suffix likewise doubles as the observation region), but
it means low-budget deltas mix two effects: (a) more structurally protected
tokens, (b) a blunter scorer for the few remaining middle slots.

## Effective-budget accounting

`derive_keep_mask` computes `budget = max(obs_window + n_sink + 4, int(T * b))` —
**protection overrides the nominal budget**. With 410+410 protection the minimum
cache is 824 tokens for every example (verified: `n_kept = 824` exactly, for all
400 examples at b=0.0625). Nominal budgets below 824/T are silently clamped:

| task | mean T | protection floor 824/T | budgets clamped (eff. budget) |
|---|---:|---:|---|
| niah_multikey_3 | 4009 | 0.206 | 0.1875, 0.125, 0.0625 → eff 0.206 |
| vt | 3876 | 0.213 | 0.1875, 0.125, 0.0625 → eff 0.213 |
| fwe | 4006 | 0.206 | 0.1875, 0.125, 0.0625 → eff 0.206 |
| qa_1 | 3073 | 0.268 | 0.25, 0.1875, 0.125, 0.0625 → eff 0.27 |

Cells marked `*` below have effective budget > nominal; the three (four for qa_1)
lowest nominal budgets are the **same cache** and are reported as a single
"protection floor" condition. Comparisons against the baseline at nominal
b ≤ 0.1875 are therefore capacity-mismatched (protected keeps ~824 tokens where
baseline keeps 256–768); matched-capacity comparisons use baseline b = 0.1875–0.25.

## Accuracy vs budget (protected = n_sink 410 / w 410; baseline = n_sink 4 / w 32)

### niah_multikey_3 (N=100)

| nominal b | eff b (prot.) | acc protected | acc baseline |
|---:|---:|---:|---:|
| 1.0 | 1.000 | 0.650 | 0.650 |
| 0.875 | 0.875 | 0.400 | 0.470 |
| 0.75 | 0.750 | 0.270 | 0.360 |
| 0.625 | 0.625 | 0.200 | 0.270 |
| 0.5 | 0.500 | 0.170 | 0.140 |
| 0.375 | 0.375 | 0.120 | 0.070 |
| 0.25 | 0.250 | 0.110 | 0.020 |
| 0.1875 | 0.206 * | 0.090 | 0.010 |
| 0.125 | 0.206 * | 0.090 | 0.000 |
| 0.0625 | 0.206 * | 0.090 | 0.000 |

### vt (N=100)

| nominal b | eff b (prot.) | acc protected | acc baseline |
|---:|---:|---:|---:|
| 1.0 | 1.000 | 0.820 | 0.820 |
| 0.875 | 0.875 | 0.810 | 0.800 |
| 0.75 | 0.750 | 0.810 | 0.810 |
| 0.625 | 0.625 | 0.830 | 0.790 |
| 0.5 | 0.500 | 0.810 | 0.800 |
| 0.375 | 0.375 | 0.800 | 0.810 |
| 0.25 | 0.250 | 0.310 | 0.820 |
| 0.1875 | 0.213 * | 0.000 | 0.810 |
| 0.125 | 0.213 * | 0.000 | 0.750 |
| 0.0625 | 0.213 * | 0.000 | 0.030 |

### fwe (N=100)

| nominal b | eff b (prot.) | acc protected | acc baseline |
|---:|---:|---:|---:|
| 1.0 | 1.000 | 0.220 | 0.220 |
| 0.875 | 0.875 | 0.220 | 0.220 |
| 0.75 | 0.750 | 0.220 | 0.230 |
| 0.625 | 0.625 | 0.210 | 0.230 |
| 0.5 | 0.500 | 0.210 | 0.210 |
| 0.375 | 0.375 | 0.230 | 0.200 |
| 0.25 | 0.250 | 0.320 | 0.180 |
| 0.1875 | 0.206 * | 0.310 | 0.150 |
| 0.125 | 0.206 * | 0.310 | 0.120 |
| 0.0625 | 0.206 * | 0.310 | 0.110 |

### qa_1 (N=100)

| nominal b | eff b (prot.) | acc protected | acc baseline |
|---:|---:|---:|---:|
| 1.0 | 1.000 | 0.740 | 0.730 |
| 0.875 | 0.875 | 0.740 | 0.700 |
| 0.75 | 0.750 | 0.720 | 0.720 |
| 0.625 | 0.625 | 0.720 | 0.700 |
| 0.5 | 0.500 | 0.730 | 0.730 |
| 0.375 | 0.375 | 0.710 | 0.740 |
| 0.25 | 0.271 * | 0.710 | 0.710 |
| 0.1875 | 0.270 * | 0.690 | 0.690 |
| 0.125 | 0.270 * | 0.690 | 0.670 |
| 0.0625 | 0.270 * | 0.690 | 0.600 |

`*` effective budget exceeds nominal (protection floor engaged; identical caches
within each task's starred rows). Full-KV (b=1.0) accuracies match the baseline
run exactly, as they must — the b=1.0 condition is protection-independent.

## ρ comparison (ρ = fraction of inputs where full-KV b=1.0 is wrong AND some b<1.0 is correct)

| task | ρ baseline (n_sink 4 / w 32) | ρ protected (410/410) | wrong@full (both runs) |
|---|---:|---:|---:|
| vt | 0.06 | 0.03 | 18 |
| fwe | 0.06 | **0.13** | 78 |
| qa_1 | 0.08 | 0.07 | 27 / 26 |
| niah_multikey_3 | 0.01 | 0.03 | 35 |

Aggregate over the three dilution-prone tasks: 23/300 = 7.7% of inputs are fixed
by some eviction level under heavy protection (vs 20/300 = 6.7% baseline). The
eviction-helps signal survives Garcia-scale protection; it is not an artifact of
light protection. Per task: fwe doubles (0.06→0.13 — the sink+recency-heavy cache
is a strong prior for a frequency task, and dropping middle noise helps more, not
less), qa_1 is stable (0.08→0.07), vt attenuates (0.06→0.03; at N=100 the 3-vs-6
difference is within binomial noise, but the direction is consistent with the
scorer-dilution confound above). MK3's ρ rises slightly (0.01→0.03) while
remaining the smallest — eviction almost never rescues MK3 inputs under either
protection regime.

## Answers to the key questions

**(1) Does NIAH-MK3 still collapse — is the failure protection-fixable?**
It still collapses; the failure is capacity-bound, not protection-fixable.
At the protection floor (824 kept tokens, eff b ≈ 0.206 — 3.3× the nominal-0.0625
baseline cache of ~256 tokens) MK3 accuracy is 0.09 vs 0.65 full-KV: an 86%
relative collapse despite 10%+10% structural protection. The lift over the
baseline floor (0.00 → 0.09) is almost entirely a capacity artifact: at matched
effective capacity (baseline b = 0.1875–0.25, i.e. 768–1024 kept tokens) the
baseline scores 0.01–0.02 vs protected 0.09–0.11, so protection buys at most
+0.07–0.09 absolute — nowhere near recovering the 0.56 lost. Moreover, at high
budgets protection *hurts* MK3 (0.40 vs 0.47 at b=0.875; 0.27 vs 0.36 at 0.75;
0.20 vs 0.27 at 0.625): 820 unconditionally protected tokens crowd out
scorer-selected middle tokens exactly where the needles live. Structural
protection cannot substitute for keeping the (middle-of-context) needle tokens.

**(2) Do the dilution-prone ρ values survive (≥5% of inputs helped)?**
Yes in aggregate and for two of three tasks: fwe 0.13 and qa_1 0.07 individually
clear 5%; vt drops to 0.03 (below 5%, within noise of baseline 0.06 at N=100).
Aggregate 7.7% ≥ 5%. Eviction still helps a non-trivial fraction of inputs under
heavy protection — indeed FWE's eviction *benefit* grows (floor accuracy 0.31 >
0.22 full-KV, +41% relative).

**(3) Verdict sentence for the paper.**
"The partition is not an artifact of light protection: under Garcia-scale
10%+10% structural protection, NIAH-MultiKey-3 still loses 86% of its full-KV
accuracy at the protection floor (0.65→0.09, with the floor cache 3.3× larger
than the light-protection budget it is compared against), while eviction
continues to repair 7.7% of dilution-prone inputs (ρ_FWE doubles to 0.13) —
heavy protection in fact *hurts* both the retrieval task at high budgets and VT
at low budgets by crowding out scorer-selected middle tokens."

## Caveats

- Single model (Qwen2.5-1.5B-Instruct), single context length (4K), N=100/task.
- The obs_window confound (scoring window = protection span) means the VT
  low-budget collapse (0.82 → 0.31 → 0.00 for nominal b ≤ 0.25) mixes protection
  crowd-out with scorer dilution; disentangling would require decoupling the two
  parameters in `ruler_sweep.py`. Either way it is evidence *against* "protection
  is all you need": a protection-heavy configuration destroys a task the
  scorer-heavy configuration preserves at the same nominal budget.
- The three starred budget rows per task are one condition, not three; do not
  average them as independent points.
