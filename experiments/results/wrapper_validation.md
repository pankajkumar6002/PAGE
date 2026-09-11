# Validation of the memory workarounds

`gate_signal_ablation.py` runs with `output_attentions=True`, retaining every
layer's full `[H,T,T]` attention: ~120 GiB for Qwen2.5-14B at 4K. That is why
the held-out cells could not simply be run, and why three wrappers were tried.
Each is checked against the stock script on the fitting cell before use.

| variant | mean $D$ on MK3 | delta vs stock |
|---|---:|---:|
| HOST_A A100, stock | 0.0448 | +0.0000 |
| HOST_B Ada, stock | 0.0412 | -0.0035 |
| HOST_B Ada, v1 retention-only | 0.0412 | -0.0035 |
| HOST_A A100, v2 SDPA rewrite | 0.0300 | -0.0148 |
| HOST_B Ada, v2 SDPA rewrite | 0.0299 | -0.0149 |
| HOST_A A100, v3 query-chunked | 0.0448 | +0.0000 |
| paper, released (A100) | 0.0450 | +0.0002 |

**Conclusions.**

* **v1 (retention-only)** calls the stock attention and slices the returned
  tensor. Faithful: identical to stock on the same host. Bounds retention but
  not the per-layer transient, so it handles 14B at 4K and not 16K.
* **v2 (SDPA rewrite)** recomputed attention. **Not faithful**: $D$ falls by
  0.0148, a 33% shift, reproducibly on both hosts. Fixing its mask handling
  changed nothing, so the cause is the SDPA output path perturbing hidden
  states and compounding across layers. Discarded; no reported cell uses it.
* **v3 (query-chunked)** chunks queries but calls the stock function unchanged.
  Faithful to +0.000029, pure floating-point reassociation.

**Hardware, separately.** Stock script, A100 vs Ada: -0.0035 on mean $D$.
The code-path effect of v2 is four times larger than the hardware effect, which
is why the 2x2 (code path x hardware) was needed to tell them apart: a single
comparison against the released numbers could not.
