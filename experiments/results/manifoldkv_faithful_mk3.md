# Faithful ManifoldKV + Ada-KV on NIAH-MK3 (closing the reviewer gap)

Reviewer ask: reproduce the *real* ManifoldKV pipeline (arXiv 2602.08343) — a
competent geometry-based scorer integrated with Ada-KV per-head budget
allocation — and test whether it rescues RULER `niah_multikey_3` (MK3) where
attention-score evictors are claimed to be capacity-bound. If it rescues MK3, the
"capacity-bound" showcase is scorer-specific.

**TL;DR verdict: (b) NO — a faithful ManifoldKV+Ada-KV does NOT rescue MK3.**
Under aggressive compression (budget <= 0.25) every scorer we tried — SnapKV,
KeyDiff, ManifoldKV (post-RoPE and pre-RoPE keys), each with uniform *and* Ada-KV
per-head allocation — collapses to ~0 on MK3. The geometry scorer does not beat a
properly-implemented per-head SnapKV at any budget; if anything it is weaker at
budget 0.5. See "Verdict" for the exact sentence and an important caveat about
the paper's *existing* SnapKV-collapse number.

---

## What ManifoldKV actually is (from the paper)

- **Score (Algorithm 1):** `s_i = || k_i - mu ||_2`, with `mu = (1/N) sum_i k_i`
  the key centroid. Keep the top-scoring (largest-distance = geometric outlier)
  tokens. L2 (not cosine) is the whole point: it preserves magnitude, so radial
  outliers (`k = alpha*mu`, alpha>1) and magnitude-encoded entities/numbers are
  not conflated ("directional collision"), which is why the paper says it beats
  KeyDiff (cosine-to-mean) on multi-key retrieval.
- **Per-head:** the centroid is per (layer, kv-head).
- **Ada-KV integration (arXiv 2407.11550):** the fixed global KV budget is
  *reallocated across heads* within a layer (head-wise adaptive) instead of a
  uniform per-head top-k. Real Ada-KV uses a safeguard floor (`floor_alpha=0.5`
  of the uniform per-head budget) plus a global top-k over the remaining
  flattened per-head scores, and stores the ragged result with a flattened cache
  + `flash_attn_varlen`. In the paper's own words the scorer decides *which*
  tokens a head keeps; Ada-KV decides *how many*.
- **Reported numbers (Llama-3.1-8B, 8K, with Ada-KV):** 3-key NIAH 92.4% vs
  KeyDiff 77.0% at 50% compression; RULER 4K-16K 95.7%; global L2 collapses at
  64K (centroid dilution), fixed by WindowedManifoldKV. No public code repo found
  for ManifoldKV itself; Ada-KV code is public (github.com/FFY0/AdaKV), KeyDiff is
  the cosine-to-mean baseline.

## Faithful implementation in this codebase

New standalone script `experiments/scripts/manifoldkv_adakv.py` (does not touch
`gated_eviction.py`). Scorers: `manifoldkv` (L2, post-RoPE keys from the cache),
`manifoldkv_pre` (L2, pre-RoPE keys = `k_proj(hidden)` with no rotary — Qwen2.5
has no q/k-norm), `keydiff` (negative cosine to centroid), `snapkv` (attention
mass of the last `obs_window` queries, per kv-head). Allocation: `uniform` per-head
top-k vs `adakv` (safeguard floor `floor_alpha=0.5` + global top-k reallocation
across the 2 kv-heads within each layer). Sink (`n_sink=4`) + recency
(`obs_window=32`) always protected, matching the other policies.

**Key correction vs the paper's failed variants.** The earlier
`manifoldkv_perlayer` / single-mask variants (see
`manifoldkv_perlayer_mk3.md`) shared ONE keep-set across heads (and often across
layers) so the DynamicCache stayed rectangular. That is the implementation gap:
it is not per-head, so it cannot express Ada-KV's ragged head-wise allocation, and
it collapsed on *every* task. Here each (layer, kv-head) gets its own keep-set.

**How ragged per-head eviction is realized (documented deviation).** Ada-KV
produces ragged per-head keep counts, which HF `DynamicCache` cannot store
rectangularly. Rather than physically prune with padding (which over-allocates
and confounds the budget), we keep the FULL cache and **mask attention per
kv-head**: we monkeypatch Qwen2's `eager_attention_forward` so that during decode
each kv-head adds `-inf` to the logits of keys outside its keep-set. This
reproduces *exactly* the logits a ragged compressed cache would produce — kept
keys retain their original RoPE rotation, evicted keys get zero weight — with
**zero over-allocation** and no position surgery. Verified: the measured mean kept
fraction equals the nominal budget to 4 decimals (e.g. nominal 0.0625 -> effective
0.0626), so there is no memory cheat inflating accuracy. This measures ACCURACY
under a logical budget, not wall-clock memory savings (which is all the reviewer
question needs). Prefill is unmasked (eviction happens after prefill); the last
prompt token is re-fed against the masked cache so no answer token benefits from
full-prompt attention.

Setup: Qwen2.5-1.5B-Instruct, RULER 4K, greedy, `max_new=48`. Raw:
`manifoldkv_faithful_mk3.jsonl` (MK3), `manifoldkv_faithful_sanity.jsonl`
(single-/2-key NIAH sanity).

## Sanity check: the scorer works on easy retrieval (harness is sound)

Single-key NIAH (`niah_single_2`, N~=15, Ada-KV allocation), accuracy vs budget:

| scorer | full | b=0.5 | b=0.25 |
|---|---:|---:|---:|
| SnapKV | 1.00 | 0.87 | 0.73 |
| ManifoldKV (post) | 1.00 | 0.94 | 0.88 |
| ManifoldKV (pre) | 1.00 | 1.00 | 0.87 |

ManifoldKV+Ada-KV holds 0.88-0.94 under 2-4x compression on single-key NIAH — as
the paper claims, and *above* SnapKV. So the geometry scorer is implemented
competently and the harness does not spuriously collapse (unlike the old
single-mask harness, which collapsed everything). This is what makes the MK3
result below trustworthy.

<!-- RESULTS_TABLE_PLACEHOLDER -->

<!-- VERDICT_PLACEHOLDER -->
