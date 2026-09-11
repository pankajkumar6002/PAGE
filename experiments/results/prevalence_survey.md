# Prevalence of the capacity-bound class in realistic workloads

tau = 0.07. Per-subtask class via analyze_realistic_workload (imported). Per-input capacity-bound = full-correct ∧ destroyed at ≥1 budget ∧ D<τ. f = pooled input-weighted share over the in-range panel. AgentLongBench (32K+) reported separately, not pooled.

| subtask | model | N | A_full | mean D | gate-open | empirical class | f (native b_min) | f (b≤0.125) | pooled |
|---|---|---:|---:|---:|---|---:|---:|---:|:-:|
| qasper | qwen14b | 30 | 0.333 | +0.0942 | 0.87 | evict-robust | 0/30 (0.000) | 0/30 (0.000) | yes |
| multifieldqa_en | qwen14b | 30 | 0.000 | +0.0578 | 0.23 | untestable(floor) | 0/30 (0.000) | 0/30 (0.000) | yes |
| trec | qwen14b | 30 | 0.700 | +0.0537 | 0.20 | evict-robust | 0/30 (0.000) | 0/30 (0.000) | yes |
| triviaqa | qwen14b | 30 | 0.967 | +0.0736 | 0.43 | evict-robust | 0/30 (0.000) | 0/30 (0.000) | yes |
| lcc | qwen14b | 96 | 0.292 | +0.0543 | 0.25 | capacity-bound | 17/96 (0.177) | 11/96 (0.115) | yes |
| repobench-p | qwen14b | 82 | 0.317 | +0.0541 | 0.26 | mixed (capacity+dilution) | 8/82 (0.098) | 5/82 (0.061) | yes |
| hotpotqa | qwen14b | 100 | 0.560 | +0.0973 | 0.86 | evict-robust | 0/100 (0.000) | 0/100 (0.000) | yes |
| passage_count | qwen14b | 95 | 0.158 | -0.0128 | 0.00 | evict-robust | 3/95 (0.032) | 2/95 (0.021) | yes |
| passage_retrieval_en | qwen15b | 100 | 0.290 | +0.1046 | 1.00 | evict-robust | 0/100 (0.000) | 0/100 (0.000) | no |
| 2wikimqa | qwen14b (MISSING) | | | | | | | |
| musique | qwen14b (MISSING) | | | | | | | |
| gov_report | qwen14b (MISSING) | | | | | | | |

## Pooled in-range capacity-bound share (Qwen2.5-14B, single-model)

- **Headline (matched floor b≤0.125): f = 18/493 = 0.0365**, Wilson 95% [0.0232, 0.0570]. Every pooled subtask is stressed to the same b=0.125 floor, so f is comparable across subtasks.
- Sensitivity (native per-subtask b_min, mixes 0.125 and 0.0625 floors): f = 28/493 = 0.0568, Wilson 95% [0.0396, 0.0809].
- Subtasks analyzed: 9/12 ingestible; gap subtasks (2wikimqa, musique, gov_report/multi_news) and AgentLongBench are added by the GPU runs in experiments/gpu/run_prevalence_gaps.sh.

