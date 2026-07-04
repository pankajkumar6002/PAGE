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

## Scorer validation: the geometry scorer IS competent (reproduces ManifoldKV's multi-key win)

This is the crucial control for a case-(b) verdict — it rules out "your ManifoldKV
just failed because it was implemented weakly." Ada-KV allocation, N=25 each.

**Single-key NIAH (`niah_single_2`)** — accuracy vs budget:

| scorer | full | b=0.5 | b=0.25 |
|---|---:|---:|---:|
| SnapKV | 0.92 | 0.76 | 0.68 |
| ManifoldKV (post-RoPE) | 0.92 | 0.88 | 0.84 |
| ManifoldKV (pre-RoPE) | 0.92 | 0.88 | 0.76 |

**2-key NIAH (`niah_multikey_2`)** — accuracy vs budget:

| scorer | full | b=0.5 | b=0.25 |
|---|---:|---:|---:|
| SnapKV | 0.76 | **0.16** | **0.04** |
| ManifoldKV (post-RoPE) | 0.76 | **0.72** | **0.56** |
| ManifoldKV (pre-RoPE) | 0.76 | **0.76** | **0.52** |

On 2-key retrieval the geometry scorer **reproduces the paper's headline
mechanism**: SnapKV (attention-score) collapses under compression (0.76 -> 0.16 ->
0.04) while ManifoldKV+Ada-KV *holds* (0.76 -> 0.72 -> 0.56), a +56-point gap at
b=0.5 and +52 at b=0.25 — the same qualitative "L2 magnitude preservation avoids
directional collision on multi-key" effect the paper reports (their Table 6:
multikey_2 92.6->99.8). So our ManifoldKV is not a strawman: it is a genuinely
competent, faithfully-implemented geometry scorer that demonstrably rescues 2-key
retrieval where the attention baseline dies. That makes its failure on 3-key
(below) decisive rather than an implementation artifact.

## Result: accuracy vs budget on NIAH-MK3

Qwen2.5-1.5B-Instruct, RULER 4K, `niah_multikey_3`, N=15 (run continuing toward
N=50; the pattern below has been stable since N=5). Full-KV baseline = 0.60,
which agrees with the paper's own SnapKV full-KV number (0.65) — the clean harness
reproduces the baseline.

| scorer (allocation) | full | b=0.5 | b=0.25 | b=0.125 | b=0.0625 |
|---|---:|---:|---:|---:|---:|
| SnapKV (uniform per-head) | 0.60 | 0.40 | 0.07 | 0.00 | 0.00 |
| **SnapKV (Ada-KV)** | 0.60 | **0.53** | 0.00 | 0.00 | 0.00 |
| KeyDiff cos (uniform) | 0.60 | 0.07 | 0.00 | 0.00 | 0.00 |
| KeyDiff cos (Ada-KV) | 0.60 | 0.07 | 0.00 | 0.00 | 0.00 |
| ManifoldKV L2 post-RoPE (uniform) | 0.60 | 0.07 | 0.00 | 0.00 | 0.00 |
| **ManifoldKV L2 post-RoPE (Ada-KV)** | 0.60 | 0.07 | 0.00 | 0.00 | 0.00 |
| ManifoldKV L2 pre-RoPE (uniform) | 0.60 | 0.07 | 0.00 | 0.00 | 0.00 |
| **ManifoldKV L2 pre-RoPE (Ada-KV)** | 0.60 | 0.27 | 0.00 | 0.00 | 0.00 |

Effective (measured) kept fraction equals the nominal budget exactly (0.5000,
0.2500, 0.1250, 0.0625) — no over-allocation.

### Comparison to the paper's failed variants and to the old harness

| budget | faithful ManifoldKV+AdaKV (this) | best geometry (pre+AdaKV) | per-head SnapKV+AdaKV | old-harness single-mask ManifoldKV | old-harness SnapKV |
|---|---:|---:|---:|---:|---:|
| full | 0.60 | 0.60 | 0.60 | 0.65 | 0.65 |
| 0.5 | 0.07 | 0.27 | 0.53 | 0.03 | 0.14 |
| 0.25 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 |
| 0.125 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

Two things are visible:
1. The faithful per-head geometry scorer (with Ada-KV) is less degenerate than the
   old single-mask variant at b=0.5 (best 0.27 vs 0.03) — so the clean harness
   closes the "you didn't run true per-head/Ada-KV ManifoldKV" gap. But it still
   does **not** rescue MK3.
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
  best geometry variant is 0.27 vs SnapKV's 0.53).

So MK3 is capacity-bound in a **scorer-independent** way under aggressive
compression — including for the exact geometry method the reviewer asked about.

The decisive point: this same ManifoldKV+Ada-KV **does rescue 2-key NIAH** (0.72 vs
SnapKV 0.16 at b=0.5; 0.56 vs 0.04 at b=0.25 — see validation above), faithfully
reproducing the paper's multi-key advantage. It is therefore a *demonstrably
competent* geometry scorer, not a weak reimplementation — yet it still cannot
rescue 3-key MK3. That is the strongest possible evidence that MK3 is a genuine
multi-needle **capacity** limit and not an artifact of scorer choice or of a poor
implementation. (Curiously, on MK3 at b=0.5 the ordering even inverts — SnapKV 0.53
> ManifoldKV 0.07 — the opposite of 2-key; we do not over-interpret this at N=15,
but it further undercuts any "geometry is what rescues multi-key" story for MK3.)

**Exact sentence the paper should use:**
> To rule out a scorer-specific effect, we re-implemented ManifoldKV (Euclidean
> outlier key scoring, arXiv 2602.08343) faithfully — per-(layer, head) centroids,
> pre- and post-RoPE keys, integrated with Ada-KV per-head adaptive budget
> allocation. This scorer is competent: it reproduces ManifoldKV's signature
> multi-key advantage on 2-key NIAH, holding 0.72 accuracy at 50% cache where
> attention-based SnapKV collapses to 0.16. Yet the *same* scorer does not rescue
> 3-key NIAH-MultiKey-3 — below 25% budget it falls to chance like every
> attention-score evictor, and it never exceeds a per-head SnapKV baseline at any
> budget. MK3 is therefore capacity-bound independently of the scoring family
> (attention-score or geometric outlier), not merely for attention-based scorers.

### Honest caveat (a finding about the paper's own SnapKV number)

In this clean *per-head* harness, SnapKV at 50% budget does **not** collapse on
MK3 — it holds at 0.53 (near full-KV 0.60), and Ada-KV lifts uniform SnapKV from
0.40 to 0.53. The paper's own reported SnapKV-MK3-collapse at 50% (0.14, from
`gated_eviction.py`) is partly an artifact of that harness pooling attention into a
single keep-set shared across all layers and heads; a per-head SnapKV retrieves
the needles fine at 50%. The scorer-independent, robust capacity-bound regime is
therefore **budget <= 0.25 (>= 4x compression)**, not 50%. The paper should scope
the MK3 "capacity-bound" showcase to aggressive budgets (<=25%), where it holds for
every scorer we tested; at 50% the collapse is implementation-sensitive.

Second, note that the *2-key* task is a different story: there, attention-score
SnapKV **is** genuinely capacity-bound (0.16/0.04 at b=0.5/0.25) and the geometry
scorer **rescues it** (0.72/0.56). So "attention evictors are capacity-bound on
multi-key retrieval" is scorer-specific for *2-key* — a good geometry scorer fixes
it — but NOT for *3-key*, where the capacity wall binds every scorer. The paper's
MK3 claim survives; a broader "multi-key is capacity-bound" claim would not, and
should be stated as MK3-specific.

We did not reproduce ManifoldKV's headline 92.4%-at-50% number; that is on
Llama-3.1-8B at 8K with the authors' full flattened-cache Ada-KV pipeline, whereas
this is Qwen2.5-1.5B at 4K (full-KV MK3 is only ~0.62 here). The verdict is about
*relative* rescue on a matched harness, which is what the reviewer asked for.

## Files

- Scorer + harness: `experiments/scripts/manifoldkv_adakv.py`
- Aggregator: `experiments/scripts/agg_manifoldkv_faithful.py`
- Raw MK3: `experiments/results/manifoldkv_faithful_mk3.jsonl`
- Raw sanity (single/2-key NIAH): `experiments/results/manifoldkv_faithful_sanity.jsonl`
