# Reproduction gate: does this environment reproduce the released logs?

Every GPU number in this round was produced in a fresh environment (Python
3.13.14 / torch 2.11.0+cu130 / transformers 5.9.0) on a machine that is not the
one that produced the released logs. Before any of it is reported, the same
configuration is re-run and diffed against the released log field by field.

Configuration: Qwen2.5-1.5B-Instruct, RULER 4K, 4 tasks, `b` in {1.0, 0.25},
N = 100, SnapKV scoring, greedy decoding, tau = 0.07. 800 rows compared against
`page-kv/experiments/results/gated_4k_qwen15b.jsonl`.

| field | HOST_A (A100-SXM4-80GB) | HOST_B (RTX 6000 Ada) |
|---|---|---|
| `correct_plain` | 800/800 | differs on ~8% |
| `correct_gated` | 800/800 | differs on ~8% |
| `gate_open` | 800/800 | 100% match |
| `n_kept_plain` | 800/800 | 100% match |
| `n_kept_gated` | 800/800 | 100% match |
| max abs delta on `drop` | **0.000e+00** | small, mean D shifts -0.0035 |
| gate decisions flipped across tau | 0 | 0 |

**HOST_A reproduces the released log exactly**, including the continuous
gate statistic. It is an A100 box, the same class as the machine that produced
the released logs, and every seed replicate, held-out ablation cell and probe
reported in this round ran there.

**HOST_B does not reproduce generations**, but the parts of the pipeline the
method consists of, gate decisions, kept-counts and tau crossings, match
exactly. Only decoded text drifts, which is the expected consequence of a
different GPU architecture under greedy decoding.

Why this matters for the seed numbers: `seed_variance.md` measures spread
across input draws. Had the runs sat on a host that shifts `D`, that shift
would be reported as seed variance. The exact match here means the spread in
that table is the input draw and nothing else.

Reproduce: `experiments/gpu/repro_gate_hostA.sh` (writes
`experiments/results/repro_hostA.jsonl`, then diffs with
`experiments/gpu/repro_diff.py`).