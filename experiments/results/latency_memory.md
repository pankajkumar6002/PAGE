# End-to-end efficiency of gated KV-cache eviction (2026-07-03)

Measured with `experiments/scripts/measure_latency.py` (reuses the
prefill/scoring/eviction/decode machinery of `gated_eviction.py`). Raw
records: `experiments/results/latency_qwen15b_4k.jsonl` and
`latency_mistral7b_4k.jsonl`.

## Methodology

- **Hardware / contention.** NVIDIA A100-SXM4-80GB, **GPU index 1**, chosen
  after polling `nvidia-smi`: at measurement time GPU 1 carried one small
  unrelated lab process (645 MiB, **25-29% background SM utilization**,
  stable over the run); GPUs 0/2/3 ran other jobs at ~95-99%. The residual
  25-29% contention means our timing numbers are mildly pessimistic:
  **throughputs are lower bounds and overhead milliseconds are upper
  bounds**. Since all configs ran interleaved under the same background
  load, *relative* comparisons are unaffected. GPU snapshots at run
  start/end are embedded in the jsonl meta records.
- **Setup.** Qwen2.5-1.5B-Instruct, RULER 4K, mixed 4-task suite
  (niah_multikey_3, vt, fwe, qa_1), batch size 1, bfloat16, greedy,
  **eager attention** (the implementation used by all accuracy experiments
  in the paper; SDPA/flash decode would be faster in absolute terms for
  every config equally). N=20 measured inputs (5 per task) after 3 warmup
  inputs. Median prompt length T=3880 tokens.
- **Timing.** `torch.cuda.synchronize` around every timed region
  (`time.perf_counter`). Decode runs a **fixed 128 steps with no EOS
  early-exit**, so every config amortizes over the same token count;
  throughput = 128 / wall-time. Each config re-runs its own prefill so
  per-config peak memory is not contaminated by another config's tensors.
- **Configs.** (a) full KV; (b) plain SnapKV b=0.125; (c) plain SnapKV
  b=0.0625; (d) gated SnapKV: evict at b=0.125 iff head-agreement drop
  >= tau=0.07, else keep full KV. The gated config always computes the
  scoring + gate signal, so its timings include the full gate overhead.
- **Memory.** `torch.cuda.max_memory_allocated`, reset via
  `reset_peak_memory_stats` (i) before prefill and (ii) again after cache
  construction / freeing of prefill temporaries, giving separate prefill
  and decode peaks. KV bytes are summed over the actual cache tensors.

## 1. Decode throughput (Qwen2.5-1.5B, 4K, median of N=20)

| Config | tok/s (median) | IQR | 128-tok decode time |
|---|---:|---:|---:|
| (a) full KV | 37.9 | [37.3, 38.2] | 3.37 s |
| (b) SnapKV b=0.125 | 37.8 | [37.4, 38.2] | 3.38 s |
| (c) SnapKV b=0.0625 | 37.5 | [37.1, 38.3] | 3.41 s |
| (d) gated, tau=0.07 (measured) | 37.7 | [37.4, 38.1] | 3.40 s |
| (d') gated, expected mix 0.75*(b)+0.25*(a) | 37.9 | -- | -- |

**Honest interpretation: at 1.5B / 4K / batch 1 there is no measurable
decode-throughput win from eviction.** All four configs are within noise
(IQRs overlap almost completely). This is expected: Qwen2.5-1.5B has only
2 KV heads (GQA), so full-KV attention at <=4K keys reads ~106 MiB of KV
per 128 steps against ~2.9 GiB of weights per step - decode is
weight-bound, not KV-bound. The efficiency payoff at this scale is
**memory, not speed** (Section 3). Throughput gains require longer
contexts, larger KV (more KV heads), or batch >1, where attention IO and
KV residency dominate; see the scaling table in Section 5.

## 2. Prefill and gate overhead (Qwen2.5-1.5B, 4K, median ms)

| Phase | Median ms | Notes |
|---|---:|---|
| Prefill (eager, 4K) | 344 | one forward, use_cache |
| SnapKV scoring (pool_score) | 1.2 | mean attention over last 32 queries |
| **Head-agreement drop (gate)** | **17.2** | max observed 25.3 |
| Eviction + cache rebuild | 3.2 | mask + index_select over 28 layers |

**Verdict on the paper's ~50 ms claim: verified as a conservative upper
bound.** Measured median is 17.2 ms (max 25.3 ms) on Qwen2.5-1.5B at 4K -
about 3x cheaper than claimed, and measured under 25-29% background GPU
contention (so if anything the true cost is lower). Caveat for the
companion claim "< 1% of prefill" (main.tex, Cost-of-the-gate): at this
small model and 4K context the gate is **~5% of the 344 ms prefill**, not
<1%; it *is* <0.5% of the end-to-end per-input latency (17 ms against
~3.7 s prefill+decode). The current implementation also builds the
Jaccard sets on CPU (`.cpu().tolist()` in `head_agreement_layer`), so
17 ms is an upper bound on an optimized implementation. Recommend the
paper phrase this as "~17 ms, <=5% of prefill and <0.5% of end-to-end
latency at 4K" or keep "~50 ms" as an explicit upper bound. Note the
50 ms figure is Qwen-1.5B-specific and should not be read as universal:
on Mistral-7B (32 heads, 496 head pairs vs 66) the same gate costs
88.5 ms (Section 4b), consistent with the paper's own O(L H^2 k)
scaling statement.

## 3. Peak GPU memory and KV-cache size (Qwen2.5-1.5B, median of N=20)

| Config | KV cache (MiB) | Peak decode alloc (MiB) | Peak prefill alloc (MiB) |
|---|---:|---:|---:|
| (a) full KV | 106.1 | 3090 | 4886 |
| (b) SnapKV b=0.125 | 13.2 (**8.0x smaller**) | 2976 | 14192 |
| (c) SnapKV b=0.0625 | 6.6 (16.1x) | 2968 | 14192 |
| (d) gated (open: 13.2 / closed: 110.5) | 13.2 median | 2976 | 14192 |
| (d') gated expected mix | 36.5 (**2.9x smaller**) | 3004 | -- |

- Measured KV bytes match the analytic formula exactly:
  layers x kv_heads x head_dim x T x 2 tensors x 2 bytes =
  28 x 2 x 128 x T x 4 = 28,672 B/token; at the median T=3880 that is
  106.1 MiB (at exactly T=4096: 112.0 MiB), and 484 kept tokens gives
  13.2 MiB. **b=0.125 keeps slightly more than 12.5%** (484/3880 = 12.5%
  plus sink+window floor) - the 8.0x reduction is as designed.
- Peak *decode* allocation is dominated by the 2944 MiB of bf16 weights;
  the full-vs-b=0.125 difference (3090 - 2976 = 114 MiB) is almost exactly
  the KV saved. At 1.5B/4K the KV is ~4% of decode footprint - again,
  material savings appear at longer T / bigger KV (Section 5).
- Peak *prefill* allocation for any scored config (14.2 GiB vs 4.9 GiB for
  the plain full-KV path) is dominated by retaining eager attention maps
  for the one-pass scorer (28 layers x [12, T, T] bf16 ~ 9.3 GiB). This is
  a cost of *any* attention-score evictor (SnapKV/H2O/PyramidKV) in
  one-pass mode, not of the gate; the codebase's `--two_pass` mode
  (flash prefill + obs-window re-forward) avoids it at long contexts.

## 4. Gated end-to-end at tau=0.07 (measured mixed sample, N=20, 5/task)

Gate-open fraction on this sample: **15/20 = 0.75**, exactly matching the
mixed-suite fraction from the N=400 tau-sweep (`tau_sweep_tau0.07.jsonl`:
0.75 overall; niah_multikey_3 0/100 open, vt/fwe/qa_1 100/100 each). Drops
on this sample separate cleanly: niah_mk3 0.036-0.056 (< tau, keep full
KV), vt 0.161-0.166, fwe 0.110-0.117, qa_1 0.216-0.242 (>= tau, evict).

| Quantity | Gated measured | Expected mix 0.75*(b)+0.25*(a) |
|---|---:|---:|
| Decode tok/s (median) | 37.7 (open 38.0, closed 37.0) | 37.9 |
| KV cache MiB (median) | 13.2 (open 13.2, closed 110.5) | 36.5 |
| Peak decode MiB | 2976 | 3004 |
| Gate overhead per input | 17.2 ms + 1.2 ms scoring + 3.2 ms rebuild | same |

So on the mixed suite the gated method delivers, in expectation, a
**2.9x KV-memory reduction** (vs 8.0x for plain SnapKV b=0.125) while - per
the accuracy experiments (`method_agnostic_matrix.md`) - recovering the
capacity-bound inputs that plain SnapKV destroys. The throughput cost of
gating vs plain SnapKV is zero at this scale (both ~= full KV), and the
total added latency per input is ~21 ms (~0.6% of end-to-end).

## 4b. Mistral-7B-Instruct-v0.3, 4K (optional second model, N=12, 3/task)

Same protocol (GPU 1, background 26% at start; eager; 128 forced decode
steps; 2 warmup inputs; median T=4102). Raw:
`latency_mistral7b_4k.jsonl`.

| Config | tok/s (median) | KV cache (MiB) | Peak decode (MiB) | Peak prefill (MiB) |
|---|---:|---:|---:|---:|
| (a) full KV | 18.9 | 512.8 | 14429 | 19712 |
| (b) SnapKV b=0.125 | 24.5 (**1.29x faster**) | 64.1 (8.0x) | 13929 | 51555 |
| expected gated mix (0.75 open) | 23.1 (1.22x) | 176 (2.9x) | 14054 | -- |

- **At 7B the throughput win is real even at 4K**: 8 KV heads give a
  512 MiB KV (analytic: 512.75 MiB - exact match), and eager decode
  attention over 4K keys is no longer negligible: SnapKV b=0.125 decodes
  **1.29x faster**; the expected gated mix at the Qwen-calibrated 0.75
  gate-open fraction is 1.22x. (Fraction not re-calibrated on Mistral
  here; the accuracy runs `gated_4k_mistral7b.jsonl` use the same
  tau=0.07.)
- Overheads on Mistral-7B (median): prefill 1898 ms (full path),
  SnapKV scoring 3.3 ms, **gate 88.5 ms**, evict+rebuild 11.0 ms. The
  gate's O(L H^2 k) cost grows with the head count (32 heads = 496 head
  pairs vs 66 on Qwen); 88.5/17.2 = 5.1x measured vs 8.1x predicted by
  L*H^2 - the paper's "extrapolates linearly" wording (Sec. Setup) is
  conservative-to-accurate. Gate is 4.7% of the Mistral prefill and
  ~1.3% of end-to-end latency (88.5 ms against 1.74 s prefill + 5.2 s
  decode on the b=0.125 path).
- Peak prefill for the scored config is 51.6 GiB (retained eager
  attentions: 32 layers x [32, 4.1K, 4.1K] bf16 ~ 33 GiB) - at 7B/4K
  the one-pass scorer is close to the A100 limit and the `--two_pass`
  path becomes the right default beyond this point.

## 5. Where the savings become material: analytic KV scaling

Analytic KV size (layers x kv_heads x head_dim x T x 2 tensors x 2 bytes,
bf16), with gated expectation at the measured 0.75 gate-open fraction:

| Model | bytes/token | T | Full KV | SnapKV b=0.125 | Gated (0.75 open) | Gated saving |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B (28L, 2KV, 128d) | 28 KiB | 4096 | 112 MiB | 14 MiB | 38.5 MiB | 73.5 MiB |
| | | 16384 | 448 MiB | 56 MiB | 154 MiB | 294 MiB |
| | | 32768 | 896 MiB | 112 MiB | 308 MiB | 588 MiB |
| Mistral-7B (32L, 8KV, 128d) | 128 KiB | 4096 | 512 MiB | 64 MiB | 176 MiB | 336 MiB |
| | | 16384 | 2.0 GiB | 256 MiB | 704 MiB | 1.34 GiB |
| | | 32768 | 4.0 GiB | 512 MiB | 1.38 GiB | 2.69 GiB |

At 4K on a 1.5B model the absolute saving (74 MiB gated) is trivially
small next to the weights. At 32K on Mistral-7B the gated saving is
~2.7 GiB *per sequence*, i.e. per-GPU batch capacity scales ~2.9x for
KV-residency-bound serving, and the KV read per decode step (the decode
bandwidth bottleneck at long T) shrinks by the same 2.9x expected factor
(8x on gate-open inputs). These are the regimes the paper should point to;
the 4K numbers here establish (i) the gate itself is ~17-21 ms, amortized
to noise, and (ii) memory reductions are exactly as the analytic formula
predicts, so the scaling column is trustworthy.

## Caveats

1. **Background contention (25-29% SM) on the measurement GPU.** All
   timings are mild upper bounds; throughputs are lower bounds. Relative
   config comparisons ran under identical conditions.
2. **Eager attention throughout**, matching the paper's accuracy
   machinery. Absolute tok/s would rise for all configs with SDPA/flash;
   the flat full-vs-evicted comparison at 4K would not change sign.
3. Decode forced to 128 steps (no EOS exit) for comparable timing; real
   generations may be shorter.
4. Single GPU model (A100-80GB), batch size 1. Batched serving would show
   larger eviction gains (KV residency limits batch size).
5. The 16K/32K rows in Section 5 are analytic, not measured; the 4K
   measured-vs-analytic agreement (106.1 vs 106.1 MiB) supports them.
