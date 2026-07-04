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

## Result: accuracy vs budget on NIAH-MK3

Qwen2.5-1.5B-Instruct, RULER 4K, `niah_multikey_3`, N=11 (run continuing toward
N=50; the pattern below has been stable since N=5). Full-KV baseline = 0.60-0.64,
which agrees with the paper's own SnapKV full-KV number (0.65) — the clean harness
reproduces the baseline.

| scorer (allocation) | full | b=0.5 | b=0.25 | b=0.125 | b=0.0625 |
|---|---:|---:|---:|---:|---:|
| SnapKV (uniform per-head) | 0.60 | 0.40 | 0.10 | 0.00 | 0.00 |
| **SnapKV (Ada-KV)** | 0.60 | **0.60** | 0.00 | 0.00 | 0.00 |
| KeyDiff cos (uniform) | 0.60 | 0.10 | 0.00 | 0.00 | 0.00 |
| KeyDiff cos (Ada-KV) | 0.60 | 0.10 | 0.00 | 0.00 | 0.00 |
| ManifoldKV L2 post-RoPE (uniform) | 0.64 | 0.09 | 0.00 | 0.00 | 0.00 |
| **ManifoldKV L2 post-RoPE (Ada-KV)** | 0.64 | 0.10 | 0.00 | 0.00 | 0.00 |
| ManifoldKV L2 pre-RoPE (uniform) | 0.60 | 0.10 | 0.00 | 0.00 | 0.00 |
| **ManifoldKV L2 pre-RoPE (Ada-KV)** | 0.60 | 0.30 | 0.00 | 0.00 | 0.00 |

Effective (measured) kept fraction equals the nominal budget exactly (0.5000,
0.2500, 0.1250, 0.0625) — no over-allocation.

### Comparison to the paper's failed variants and to the old harness

| budget | faithful ManifoldKV+AdaKV (this) | best geometry (pre+AdaKV) | per-head SnapKV+AdaKV | old-harness single-mask ManifoldKV | old-harness SnapKV |
|---|---:|---:|---:|---:|---:|
| full | 0.64 | 0.60 | 0.60 | 0.65 | 0.65 |
| 0.5 | 0.10 | 0.30 | 0.60 | 0.03 | 0.14 |
| 0.25 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 |
| 0.125 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

Two things are visible:
1. The faithful per-head geometry scorer (with Ada-KV) is *much* less degenerate
   than the old single-mask variant at b=0.5 (0.10-0.30 vs 0.03) — so the clean
   harness closes the "you didn't run true per-head/Ada-KV ManifoldKV" gap. But it
   still does **not** rescue MK3.
2. Below 25% budget, *every* method — attention (SnapKV), cosine (KeyDiff),
   L2 (ManifoldKV, both RoPE variants), uniform and Ada-KV — is at 0.00.

## Verdict: case (b) — faithful ManifoldKV+Ada-KV does NOT rescue MK3

The reviewer's alternative ("a competent geometry scorer rescues MK3, so the
capacity-bound claim is scorer-specific") is **not supported**. A faithful
ManifoldKV (Euclidean-outlier L2 key scoring, per (layer, kv-head) centroid,
tried on both post-RoPE and pre-RoPE keys) integrated with Ada-KV per-head
adaptive budget allocation (floor_alpha=0.5 + global head-wise top-k):
- collapses to chance (0.00) at budget <= 0.25, exactly like SnapKV/KeyDiff; and
- never beats a properly-implemented per-head SnapKV at any budget (at b=0.5 the
  best geometry variant is 0.30 vs SnapKV's 0.60).

So MK3 is capacity-bound in a **scorer-independent** way under aggressive
compression — including for the exact geometry method the reviewer asked about.
The geometry scorer works fine on *single-key* NIAH (0.88-0.94 at b=0.5-0.25,
above SnapKV), so its MK3 failure is a genuine multi-needle capacity limit, not a
weak-scorer artifact.

**Exact sentence the paper should use:**
> To rule out a scorer-specific effect, we re-implemented ManifoldKV (Euclidean
> outlier key scoring, arXiv 2602.08343) faithfully — per-(layer, head) centroids,
> pre- and post-RoPE keys, integrated with Ada-KV per-head adaptive budget
> allocation — and evaluated it on NIAH-MultiKey-3. It does not rescue the task:
> below 25% cache budget it collapses to chance like every attention-score
> evictor, and it never exceeds a per-head SnapKV baseline at any budget. MK3 is
> therefore capacity-bound independently of the scoring family (attention-score or
> geometric outlier), not merely for attention-based scorers.

### Honest caveat (a finding about the paper's own SnapKV number)

In this clean *per-head* harness, SnapKV at 50% budget does **not** collapse on
MK3 — it holds at 0.60 (= full-KV), and Ada-KV even lifts uniform SnapKV from 0.40
to 0.60. The paper's own reported SnapKV-MK3-collapse at 50% (0.14, from
`gated_eviction.py`) is partly an artifact of that harness pooling attention into a
single keep-set shared across all layers and heads; a per-head SnapKV retrieves
the needles fine at 50%. The scorer-independent, robust capacity-bound regime is
therefore **budget <= 0.25 (>= 4x compression)**, not 50%. The paper should scope
the MK3 "capacity-bound" showcase to aggressive budgets (<=25%), where it holds for
every scorer we tested; at 50% the collapse is implementation-sensitive.

We did not reproduce ManifoldKV's headline 92.4%-at-50% number; that is on
Llama-3.1-8B at 8K with the authors' full flattened-cache Ada-KV pipeline, whereas
this is Qwen2.5-1.5B at 4K (full-KV MK3 is only ~0.62 here). The verdict is about
*relative* rescue on a matched harness, which is what the reviewer asked for.

## Files

- Scorer + harness: `experiments/scripts/manifoldkv_adakv.py`
- Aggregator: `experiments/scripts/agg_manifoldkv_faithful.py`
- Raw MK3: `experiments/results/manifoldkv_faithful_mk3.jsonl`
- Raw sanity (single/2-key NIAH): `experiments/results/manifoldkv_faithful_sanity.jsonl`
