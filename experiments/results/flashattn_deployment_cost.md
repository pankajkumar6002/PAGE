# Deployment cost of PAGE's attention-materialization (two-pass) gate at scale (2026-07-04)

Measures the cost of a real deployment constraint: the PAGE gate scores the
head-agreement drop **D** from *prefill
attention weights* (per-head, per-key), which fused FlashAttention / SDPA never
materialize. The paper's escape is a **two-pass** mode:

- **pass 1** — SDPA (or flash) prefill, `output_attentions=False`, gives the
  length-`T` KV cache but **no** attention weights;
- **pass 2** — an **eager re-forward of the last `w=32` queries** against all `T`
  keys with `output_attentions=True`, producing a `[B, H, w, T]` attention map
  **per layer**, from which `D` is scored (`compute_drop_from_attentions`).

Script: `experiments/scripts/measure_flashattn_gate.py` (reuses
`compute_drop_from_attentions` / `head_agreement_layer` from
`gated_eviction.py` — the identical scoring path used in the accuracy
experiments). Raw: `experiments/results/flashattn_gate_qwen32b.jsonl`,
`flashattn_gate_qwen15b.jsonl`. bf16, batch 1, `torch.cuda.synchronize` around
every timed region, **N≥10 medians** (15 for Qwen-1.5B / synthetic, 12 for the
capacity-limited Qwen-32B run) after 2–3 warmup iters.

> **Hardware / contention caveat.** A100-SXM4-80GB, shared cluster. During this
> run the four cards carried heavy, *fluctuating* lab training jobs (free memory
> swung between ~2 GiB and ~68 GiB minute-to-minute). Two consequences: (i) the
> 32B (61 GiB weights) two-pass run could only be launched when a card
> transiently exposed enough free memory, so its longer-context rows are
> capacity-limited (see notes); (ii) absolute millisecond timings are **upper
> bounds** — in particular the *as-implemented* drop cost includes per-head
> `.cpu().tolist()` GPU→CPU syncs that serialize behind the busy GPU. We
> therefore report the drop cost **two ways**: as-implemented (deployment-real
> but contention-inflated) and the **pure-CPU algorithmic floor** (the pairwise
> Jaccard only), which is contention-independent and is what a batched-transfer
> implementation achieves. The *scaling exponents and memory formulas below are
> contention-independent* and are the load-bearing results.

Models (query-head count H drives the gate cost):

| Model | Layers L | Query heads H | KV heads | head_dim | head-pairs H(H-1)/2 |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B-Instruct | 28 | 12 | 2 | 128 | 66 |
| Mistral-7B-Instruct-v0.3 | 32 | 32 | 8 | 128 | 496 |
| Qwen2.5-32B-Instruct | 64 | 40 | 8 | 128 | 780 |
| Llama-70B-class (extrapolated) | 80 | 64 | 8 | 128 | 2016 |

---

## 1. Two-pass gate cost vs context (ms, median of N=15, synced)

Gate cost = eager pass-2 re-forward of the `w=32` window **+** head-agreement-drop
computation. The two components scale differently:

- **re-forward** (GPU): O(L·H·w·T) compute → **~linear in T** (a full model
  forward over 32 tokens, attending to all `T` keys, per layer).
- **drop** (CPU): O(L·H²·k), **independent of T** (depends only on the top-`k`
  sets and the head count) — confirmed <3% variation 4K→32K.

### Qwen2.5-1.5B (L=28, H=12), measured

| T | re-forward ms | drop ms (as-impl) | drop ms (algo floor) | gate ms (as-impl) | SDPA prefill ms |
|---:|---:|---:|---:|---:|---:|
| 4096  | 48.9 | 141 | ~2 | 192 | 365 |
| 16384 | 89.4 | 148 | ~2 | 234 | 1501 |
| 32768 | 128.1 | 124 | ~2 | 251 | 3149 |

Re-forward fit: **0.00274·T + 40 ms** (per-token 2.7 µs). The drop's
as-implemented 124–148 ms is almost entirely the L·H=336 serialized per-head
GPU→CPU transfers under contention (the algorithmic Jaccard floor for H=12 is
~2 ms; the prior lighter-contention run in `latency_memory.md` measured 17 ms).

### Qwen2.5-32B (L=64, H=40), measured

| T | re-forward ms | drop ms (as-impl) | drop ms (algo floor) | gate ms (as-impl) | SDPA prefill ms | prefill peak | scoring peak |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096  | **144.8** | **203.5** | ~87 | **349.7** | 1914 | 64753 MiB | 64468 MiB |
| 16384 | OOM* | (87, T-indep.) | ~87 | -- | OOM* | -- | -- |

\* **16K OOM'd on the shared 80 GiB A100**: 61 GiB weights + full-length KV +
the eager `w×T` attention transient + HF full-sequence logits exceeded the free
budget even from a fresh 68 GiB window (the concurrent lab jobs re-expanded
during the 21 s model load). This is *itself* direct evidence of the deployment
limitation — a 32B two-pass gate barely fits at 4K and does not fit at 16K on a
single contended 80 GiB card. The drop cost is T-independent (§3: 87 ms
algorithmic floor at every T), and the 16K/32K memory is analytic and validated
in §2. At **4K the gate is 350 ms** (145 ms eager re-forward over 64 layers +
204 ms as-implemented drop), i.e. **18% of the 1.9 s prefill** — vs ~5% on
Qwen-1.5B; the fraction grows with head count exactly as O(L·H²) predicts.

---

## 2. Peak memory: pass-2 scoring vs SDPA prefill

The main memory cost is the **`w × T` attention materialization**: pass-2
retains, for **all L layers at once** (`output_attentions=True`), a `[H, w, T]`
bf16 tensor. Analytic size **L·H·w·T·2 bytes** (Qwen-1.5B: 672 MiB at 32K). The
measured Qwen-1.5B scoring-pass peak decomposes consistently as weights (2.9 GiB)
+ length-`T` KV + this materialization (e.g. 32K: 2.9 GiB + 0.94 GiB KV +
0.66 GiB attn + overhead ≈ 4843 MiB measured):

| Model (L, H) | attn-materialization @4K | @16K | @32K |
|---|---:|---:|---:|
| Qwen-1.5B (28, 12) | 84 MiB | 336 MiB | 0.66 GiB |
| Mistral-7B (32, 32) | 256 MiB | 1.0 GiB | 2.0 GiB |
| Qwen-32B (64, 40) | 640 MiB | 2.5 GiB | 5.0 GiB |
| 70B-class (80, 64) | 1.25 GiB | 5.0 GiB | **10.0 GiB** |

Measured peaks (Qwen-1.5B): SDPA prefill 4265 / 8198 / 13442 MiB at 4K/16K/32K;
pass-2 scoring 3217 / 3914 / 4843 MiB. Measured peaks (**Qwen-32B @4K**): SDPA
prefill **64753 MiB**, pass-2 scoring **64468 MiB** — i.e. weights (61 GiB) +
KV + the 640 MiB attention transient already sit at ~63 GiB at only 4K, leaving
no room on an 80 GiB card once the length-`T` KV grows (16K prefill OOM'd, §1).
The materialization scales **linearly in both H and T** and is a *transient*
per-sequence allocation the pass must hold on top of weights + full-length KV.

---

## 3. Head-count scaling of the gate — is it O(H²)?

The drop computation is the head-agreement Jaccard over **all H(H-1)/2 head
pairs**, per layer, over top-`k` sets → **O(L·H²·k)**. Timing it model-free
(shape-only; the top-`k` set size is fixed so timing depends only on the shape,
exactly reproducing the accuracy-path code):

**L fixed = 64, sweep H, per-pair Jaccard cost is ~constant ≈ 0.11 ms (H≥32):**

| H | 8 | 12 | 16 | 24 | 32 | 40 | 48 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| head-pairs | 28 | 66 | 120 | 276 | 496 | 780 | 1128 | 2016 |
| drop ms | 1.7 | 4.9 | 11.6 | 29.3 | 54.9 | 87.9 | 130.5 | 232.3 |

Log-log fit in H: **exponent ≈ 2.0–2.4** (2.16 over H≥16, 2.36 over the full
H=8..64 range; per-pair cost ~constant → the total tracks H(H-1)/2 almost
exactly). **T-independent** (H=40: 90/88/88 ms at 4K/16K/32K).

**Across the real models at their native (L, H)** (pure-Jaccard floor, T=16K):

| Model | L | H | L·H² | drop ms (algo floor) |
|---|---:|---:|---:|---:|
| Qwen-1.5B | 28 | 12 | 4 032 | 2.1 |
| Mistral-7B | 32 | 32 | 32 768 | 27.3 |
| Qwen-32B | 64 | 40 | 102 400 | 87.3 |
| 70B-class | 80 | 64 | 327 680 | 291.2 |

- **Per-layer-normalized exponent in H (drop/L vs H): 2.34.**
- **Model-to-model exponent in H (native L): 2.97** — steeper than quadratic
  because L *also* grows with H across these models (the O(L·H²) product grows
  ~cubically along the real model-size axis).

**Verdict:** the measured gate cost scales **quadratically in head count**
(≈ O(H²), per-layer exponent 2.34; ≈ O(H^3) along the real model axis where L
grows with H), exactly matching the paper's stated **O(L·H²·k)** and **directly
contradicting the paper's "extrapolates linearly" wording.** The two statements
in the paper are mutually inconsistent; the O(L·H²·k) one is correct and
"extrapolates linearly" must be struck.

---

## 4. 70B-class extrapolation (80 layers, 64 heads) @ 32K

- **Weights:** 70B bf16 = **130 GiB** → does not fit one 80 GiB A100; requires
  ≥2-way tensor parallelism. The eager pass-2 with `output_attentions` must then
  gather per-head attention across the TP shards.
- **Full-length KV @32K:** 10 GiB (8 KV heads).
- **Pass-2 attention materialization @32K:** **10 GiB** transient
  (80·64·32·32768·2 B), held for all layers simultaneously — on top of weights +
  KV.
- **Drop compute:** ~291 ms algorithmic floor (H=64, L=80); the as-implemented
  per-head-transfer variant is several× that and CPU/PCIe-serialized.
- **Re-forward:** the eager `w×T` pass over 80 layers of a 130 GiB model; even
  at the 2.7 µs/token/layer-scaled rate this is hundreds of ms of GPU time that
  cannot overlap the fused decode kernels.

**Viability in a paged-FlashAttention serving stack (vLLM/TGI-style): no.** Such
stacks (i) store KV in **non-contiguous paged blocks** and (ii) compute attention
with **fused FlashAttention kernels that never emit the score matrix**. PAGE's
gate needs that score matrix. The two-pass escape requires an **eager,
non-paged, non-fused** attention re-forward that (a) needs the KV gathered into
contiguous tensors, (b) materializes a **10 GiB** per-sequence attention transient
at 70B/32K that paged attention exists specifically to avoid, and (c) feeds a
**CPU-side O(L·H²·k) ≈ 0.3 s** drop computation that stalls the pipeline — all
per request, at prefill time. It is not deployable as an online per-request
operation at 70B/long-context scale in a fused/paged stack.

---

## Honest limitation statement the paper should adopt

> PAGE's gate is scored from per-head, per-key prefill attention weights, which
> fused FlashAttention/paged-attention kernels do not materialize. In such a
> serving stack the gate requires a two-pass workaround: a standard fused prefill
> followed by a **separate eager re-forward of the last w=32 queries with
> `output_attentions=True`**. This re-forward (i) is incompatible with paged,
> non-contiguous KV and fused attention kernels, (ii) materializes an
> **L·H·w·T** attention transient — **5 GiB at 32B/32K, 10 GiB at 70B/32K** — on
> top of weights and the full-length KV, and (iii) feeds a head-agreement
> computation whose cost is **O(L·H²·k)**, i.e. **quadratic in the head count**
> (measured per-layer exponent ≈2.3, T-independent: ~90 ms at H=40/L=64,
> extrapolating to ~0.3 s at H=64/L=80). Measured end-to-end, the two-pass gate
> is **350 ms on Qwen-32B at 4K (18 % of prefill)** vs ~190 ms on Qwen-1.5B
> (5 %); on the shared 80 GiB A100 the 32B two-pass already OOM'd at 16K. The
> gate therefore does **not** "extrapolate linearly"; its cost grows with L·H²
> and its memory with L·H·T.
> This is acceptable for **offline / single-stream / research prefill** (the
> setting of all accuracy experiments here) but is **not viable as an online
> per-request operation in a high-throughput paged-FlashAttention deployment**
> at large head-count, long-context scale. The paper should (a) replace
> "extrapolates linearly" with the O(L·H²·k) statement, (b) scope the efficiency
> claims to the offline/prefill setting, and (c) note the two-pass memory and
> paged-attention incompatibility as an explicit deployment limitation.

### Notes
- Drop timing is model-free/shape-only and reproduces the exact accuracy-path
  code (`head_agreement_layer`); the top-`k` set size is fixed so shape-timing =
  real timing for the Jaccard.
- The as-implemented drop cost carries an avoidable L·H per-head GPU→CPU transfer
  overhead (contention-amplified here); a batched single transfer collapses it to
  the algorithmic floor. This is an implementation note, not a defense of the
  O(L·H²·k) fundamental cost, which stands.
