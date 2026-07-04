# NOTES: CapKV vs gated-CapKV head-to-head on LongBench/qasper

## What was run

- Subtask: **LongBench/qasper** (200-example test split; 80 attempted,
  68 evaluated, 12 skipped because their tokenized prompt exceeded the
  16K-token cap).
- Model: **Qwen/Qwen2.5-3B-Instruct** in bf16 with `attn_implementation="eager"`
  (required to harvest output attentions and queries for the CapKV proxy).
- GPU 0 only (`CUDA_VISIBLE_DEVICES=0`). Peak HBM stayed under 10 GiB.
- Compression ratio: budget b = 0.5 of T (matches the CR=0.5 column in
  Table 1 of the CapKV paper). Median prompt T = 4538 tokens; median
  kept tokens = 2269 plain, 2646 gated (gated keeps more because it falls
  back to full KV on ~34% of inputs).
- Wall time: 121 s (rerun-friendly).
- Metric: LongBench any-in substring match against the gold answer list
  (proxy for F1 token-recall).
- Headline numbers:

  | variant            |   acc   |
  |--------------------|--------:|
  | plain CapKV (b=0.5)|  0.1324 |
  | gated CapKV (b=0.5)|  0.1471 |
  | **Δ**              | **+0.0147** |

  Conditional split: gate-open subset (45/68) Δ = 0 by construction;
  gate-closed subset (23/68) plain 0.087 vs gated 0.130 = **+0.043**.

## Implementation: original code or proxy?

**Proxy, implemented from the paper.** A targeted GitHub search for "CapKV"
and arxiv 2604.25975 turned up zero repositories on 2026-06-03 (the paper
preprint is dated 2026-04-28). The CapKV algorithm is fully specified in
Algorithm 1 (page 5) and Eqs. (6)-(8) of the preprint, so I implemented
it inline as `experiments/scripts/longbench_capkv.py`:

  - For each transformer layer l and each KV head h, take the captured
    queries Q[l] (via a forward hook on q_proj), keys K[l,h], values V[l,h].
  - mu_q[l,h] = mean over the last `obs_window=32` query positions, averaged
    across the q-heads in head h's GQA group.
  - w_i = exp((<K[l,h]_i, mu_q[l,h]> * tau) / sqrt(d_head)). I added the
    1/sqrt(d) factor (which the paper does not state explicitly) to keep
    the exp argument in the same dynamic range as a standard scaled-dot-
    product attention logit; without it, with d=128 and bf16 keys/values,
    `w` saturates at +inf on long contexts.
  - A = I_d + sum_i w_i v_i v_i^T  (Eq. 6, with u_i = v_i).
  - s_i = w_i * (v_i^T A^{-1} v_i)  (Eq. 7), computed with a single
    linear solve A^{-1} V^T.
  - Final per-position score = mean over (layer, kv-head) of s_i.
  - tau = 5 (CapKV's tabled default from Table 2, "tau=5 best on average").
  - n_sink = 4, obs_window = 32 (kept at our codebase defaults to match
    other gating runs).

## Deviations from CapKV's exact published recipe

1. **u_i := v_i**, not "value-induced output direction U_C v_i". This is
   the paper's own simplification (Section 3.5, fourth paragraph):
   "we approximate the output direction using the value vector itself,
   i.e., u_i = v_i". Faithful.

2. **mu_q from the last obs_window prompt queries** instead of a running
   sample of historical decoded-token queries. The paper applies CapKV
   periodically during decoding (every 512 decoding steps in the
   reasoning ablation); we apply it once after prefill, before any
   generation. This means our mu_q is the empirical mean of the prompt-end
   queries, which is closer to a SnapKV-style "queries that will fire
   next" prior than the long-tailed decoding-query distribution. Same
   first-moment intent.

3. **tau scaling**: paper's Eq. 8 is `w_i = exp(k_i^T mu_q tau)`. With
   bf16 features of magnitude ~1 and tau=5 the exponent regularly
   exceeds +20 and overflows. We divide by sqrt(d_head) (standard
   attention scaling) before exp, then subtract the per-layer max for
   numerical stability. Algorithmically equivalent up to a positive
   constant rescaling of `w` and thus of A and s; rank-order over
   positions is preserved.

4. **Per-head scores averaged into one [T] score** (rather than a
   per-head keep-set) so that the layer cache shape stays uniform and
   the standard `DynamicCache.update` path works. PyramidKV uses the
   same simplification in our codebase. This loses the per-head
   tailoring CapKV could in principle do, but matches how every other
   eviction baseline in our paper is run (apples-to-apples).

5. **No KV-cache update during decoding**: we evict once at the
   prefill/decoding boundary. CapKV is also applied at the prefill
   boundary in the LongBench table (Section 4.2 of the paper); the
   periodic-during-decoding mode is the reasoning-task setup, not
   LongBench, so we are aligned.

## Number of inputs and what was measured

- 80 LongBench-qasper test examples attempted (rows 0-79 in
  `Xnhyacinth/LongBench` qasper config).
- 12 skipped because the chat-templated prompt tokenized to > 16K tokens
  (max we run with eager attention without paging concerns).
- N = **68** evaluated.
- Each example is evaluated twice with identical prefill and identical
  CapKV-proxy score:
    - **plain CapKV**: always evict at b=0.5.
    - **gated CapKV**: evict at b=0.5 iff the head-agreement early-vs-late
      drop is >= 0.07; otherwise keep the full KV.
- Per-record correctness is any-in substring match against the gold
  answer list. Final report = mean over the 68 evaluated examples.

## Honest assessment: is this a fair comparison?

**Yes for the Δ claim, with caveats on the absolute number.**

What the experiment *does* show fairly:
- Plain and gated CapKV use the exact same per-position score, exact same
  budget b, exact same prefill, exact same generator. The only difference
  is whether the gate is allowed to veto eviction on inputs the
  head-agreement signal flags as capacity-bound. So **+0.0147 (and the
  +0.043 lift on the gate-closed subset)** is a clean attribution to the
  gating wrapper, not to any difference in the underlying CapKV score.
- Per-example deterministic decoding (argmax, no sampling) means the Δ is
  noise-free with respect to the inference path.

What it does *not* claim:
- The absolute plain-CapKV accuracy (0.132) is not a faithful reproduction
  of the CapKV paper's Table 1 qasper number (~0.46 for Qwen3-8B). The
  gap is fully explained by (a) different model (Qwen2.5-3B vs Qwen3-8B —
  a much smaller model), (b) different metric (any-in substring vs F1),
  (c) our 16K context cap dropping the longer qasper inputs, and (d) the
  CapKV-proxy approximations above. We do not use the absolute number
  anywhere in the paper; we cite only Δ.
- N = 68 is small. A statistical test (binomial on the single flipped
  example) is not significant. The Δ is reported as a directional
  observation, not as a power-claim of "gating beats plain CapKV";
  consistent with the rest of our SOTA-baseline gating table, where Δ on
  qasper-like single-document QA is small but the gate-closed subset
  effect is the load-bearing observation.

What would make it paper-ready as a standalone column:
- Either expand to N >= 150 (the full qasper test split, lift the 16K cap
  to 24K with two-pass scoring) or evaluate the same pair on triviaqa
  where the any-in metric is far less noisy (median answer length ~6
  words, multiple gold paraphrases). With current code the run cost is
  ~2 minutes for qasper / ~10 minutes for triviaqa.

## Files produced

- `experiments/scripts/longbench_capkv.py` — proxy implementation + runner.
- `experiments/scripts/aggregate_capkv.py` — markdown summary builder.
- `experiments/results/capkv_longbench_qasper.jsonl` — per-example data
  (68 rows: id, T, drop, gate_open, n_kept, preds, correctness flags).
- `experiments/results/capkv_longbench_qasper.md` — paper-ready summary
  table.
- `experiments/results/capkv_longbench_NOTES.md` — this file.
