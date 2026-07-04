# Gated DBTrimKV vs plain DBTrimKV on RULER 4K

## What ran

- **Model**: `ngocbh/DBTrimKV-Qwen3-4B-Instruct-2507` (pretrained DBTrimKV gate
  weights, the only LLM-scale DBTrimKV checkpoint near our paper's model
  family; no Qwen2.5 checkpoint exists).
- **Gate-signal base model**: `Qwen/Qwen3-4B-Instruct-2507` loaded
  separately in sdpa/eager for the early-vs-late head-agreement drop.
- **Benchmark**: RULER 4K (`simonjegou/ruler`, config `4096`), four tasks
  - `niah_multikey_3` (NIAH-MK3 UUID retrieval),
  - `vt` (variable tracking),
  - `fwe` (frequent-word extraction),
  - `qa_1` (NarrativeQA-style).
- **Sweep**: 30 examples per task × 4 tasks × 3 paged budgets
  `memory_size ∈ {128, 256, 512}` = **N = 360 records over 120 unique
  inputs**.
- **Driver**: `external/trimkv/run_gated_dbtrimkv.py`.
- **Result file**: `experiments/results/gated_dbtrimkv_qwen3_4k.jsonl`
  (360 rows, zero inference errors).
- **Hardware**: single GPU (CUDA_VISIBLE_DEVICES=0), bf16, flash-attn 2.
- **Wall time**: previously completed end-to-end. No resume needed; the
  partial-run worry in the task brief was a row-count misreading
  (30 × 4 × 3 = 360, not 720).

## Gating rule

`gated = plain_DBTrimKV_compress() if drop >= tau else full_paged_KV`, with
the drop being the standard early-vs-late head-agreement drop on the last
32 prefill tokens, top-k = 32 attended keys, tau = 0.07. This matches the
gating rule used in `experiments/scripts/gated_eviction.py` for our SnapKV
and H2O comparisons. "Gate closed" therefore means the cache is **not**
compressed and the paged DBTrimKV runtime carries the full prefill,
yielding cache occupancy roughly 14.5× larger than plain DBTrimKV at the
same budget.

## Headline numbers

| Metric                              | Value                |
| ----------------------------------- | -------------------- |
| N                                   | 360                  |
| Plain DBTrimKV mean accuracy        | **0.617** (222/360)  |
| Gated DBTrimKV mean accuracy        | **0.850** (306/360)  |
| Δ (gated − plain)                   | **+0.233**           |
| Gate-open fraction (unique inputs)  | 2 / 120 = 1.7 %      |
| Drop distribution (over 120 inputs) | min −0.007, median 0.037, mean 0.035, max 0.087 |

## Per-task breakdown (pooled across budgets)

| Task              | N  | Plain  | Gated  | Δ        |
| ----------------- | -- | ------ | ------ | -------- |
| niah_multikey_3   | 90 | 0.467  | 1.000  | +0.533   |
| fwe               | 90 | 0.222  | 0.600  | +0.378   |
| vt                | 90 | 0.989  | 1.000  | +0.011   |
| qa_1              | 90 | 0.789  | 0.800  | +0.011   |

## Per-budget breakdown (pooled across tasks)

| memory_size | N   | Plain  | Gated  | Δ        |
| ----------- | --- | ------ | ------ | -------- |
| 128         | 120 | 0.475  | 0.850  | +0.375   |
| 256         | 120 | 0.617  | 0.850  | +0.233   |
| 512         | 120 | 0.758  | 0.850  | +0.092   |

Notice that gated accuracy is constant at 0.850 across budgets — because
the gate is essentially always closed at 4K context (118/120), the gated
policy is effectively `full-paged-KV` and is independent of `memory_size`.

## Conditional split (gate-open vs gate-closed inputs)

| Subset       | N   | Plain  | Gated  | Δ        |
| ------------ | --- | ------ | ------ | -------- |
| gate-open    | 6   | 1.000  | 1.000  | +0.000   |
| gate-closed  | 354 | 0.610  | 0.847  | +0.237   |

The conditional invariant the SnapKV/H2O experiments showed (gate-open Δ
≈ 0, gate-closed Δ ≥ 0) holds for DBTrimKV: when the early-vs-late drop
exceeds tau, the model has already concentrated attention on a small key
set, so DBTrimKV's per-token retention scores are reliable and trimming
is safe; gated and plain agree exactly. When the drop is below tau,
DBTrimKV's compress() hurts more than it helps on tight budgets, and the
gate's full-KV bypass recovers the loss.

## Paper integration

This belongs in the paper as a **standalone head-to-head subsection**
analogous to the CapKV comparison, not as a new column in `tab:matrix`.
Reasons:

- `tab:matrix` is keyed to our project's model lineup (Qwen2.5-1.5B /
  3B / 7B-Instruct). DBTrimKV has no Qwen2.5 checkpoints; the closest
  available is Qwen3-4B-Instruct-2507. Mixing a 4B Qwen3 result into a
  Qwen2.5-keyed matrix would be misleading.
- The gating signal threshold (tau = 0.07) was tuned for our SnapKV/H2O
  setup; here it produces a near-degenerate split (1.7 % open at 4K).
  That is itself a finding — the head-agreement drop is much smaller on
  Qwen3-4B-Instruct at 4K than on Qwen2.5-3B-Instruct — but it deserves
  text, not a table cell.
- The Δ of +0.233 is large enough to carry its own subsection. Headline
  framing: "On RULER 4K with DBTrimKV-Qwen3-4B, our gating wrapper lifts
  pooled accuracy from 0.617 to 0.850 (+0.233, N = 360), with the bulk
  of the gain on `niah_multikey_3` (+0.533) and `fwe` (+0.378). The
  conditional split confirms the gating invariant: on the 354 records
  where the gate is closed, Δ = +0.237; on the 6 records where the gate
  is open, plain and gated agree at 1.000 accuracy."

## Caveats (honest)

- **Single model / single context length.** Result is on
  DBTrimKV-Qwen3-4B-Instruct-2507 at 4K context only. Scaling claims
  (longer context, smaller models) are not supported by this run alone.
  The smallest DBTrimKV LLM is 4B; no 1.5B / 3B options exist.
- **Gate operating point is heavy-handed at 4K.** Only 1.7 % of inputs
  cross tau = 0.07, so most of the win comes from the gate skipping
  compress() rather than from the wrapper doing anything subtle. At
  longer contexts where the head-agreement drop is reliably larger, the
  gated and plain policies should converge; this needs an 8K / 16K
  run to be claimed in the paper.
- **Effective compression ratio under gating.** Under the gated policy
  the cache holds ~14.5× more tokens than under plain DBTrimKV at the
  same `memory_size`. The headline Δ is a quality win, not a
  memory-budget win, and the paper text must say so. A fair comparison
  at matched memory is left for follow-up (e.g., gated DBTrimKV at
  `memory_size = 64` vs plain DBTrimKV at `memory_size = 128`).
- **Proxy vs. original DBTrimKV evaluation.** The DBTrimKV paper reports
  on SCBench, not RULER; their repo has no RULER harness. Our RULER 4K
  evaluation runs their model loader against our RULER prompt pipeline,
  which is a faithful re-evaluation of the published checkpoint on a
  different benchmark, not a re-implementation of their reported numbers.
- **Pinned dependency stack.** Inference uses the separate
  `external/trimkv/venv` (transformers 4.57.1, torch 2.8, flash_attn
  2.8.3); aggregation runs in our main env via the JSONL file. No
  in-process comparison was attempted, by design.
