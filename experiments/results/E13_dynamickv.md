# E13: DynamicKV adaptive-budget head-to-head vs the PAGE gate

Arms at matched memory: **plain** = per-layer adaptive budget; **uniform** = same scorer, uniform per-layer budget (isolates the adaptivity); **gated** = PAGE gate over plain. `uniform` vs `plain` = what adaptivity buys; `plain` vs `gated` = residual admission value.

## dynamickv_qwen3_4k  (N=50 inputs/ task; budgets [1.0, 0.5, 0.25, 0.125, 0.0625])

### niah_multikey_3 (capacity-bound)

| budget | plain (adaptive) | uniform | gated | gated−plain | plain kept |
|---:|---:|---:|---:|---:|---:|
| 1.0 | 1.000 | 1.000 | 1.000 | +0.000 | 1.000 |
| 0.5 | 0.940 | 0.960 | 1.000 | +0.060 | 0.499 |
| 0.25 | 0.060 | 0.120 | 1.000 | +0.940 | 0.250 |
| 0.125 | 0.000 | 0.000 | 1.000 | +1.000 | 0.125 |
| 0.0625 | 0.000 | 0.000 | 1.000 | +1.000 | 0.062 |

- full-cache (b=1.0) MK3 accuracy: 1.000
- **P1** (adaptive alloc does NOT recover MK3 @ b=0.0625): plain=0.000 < 0.2 → HOLDS
- **P2** (gate adds value: gated−plain ≥ +0.3 @ b≤0.125): max=+1.000 → HOLDS

### Dilution-prone tasks (gate should do no HARM: gated ≥ plain)

| task | budget | plain | uniform | gated | gated−plain |
|---|---:|---:|---:|---:|---:|
| fwe | 0.5 | 0.620 | 0.620 | 0.660 | +0.040 |
| fwe | 0.25 | 0.620 | 0.600 | 0.660 | +0.040 |
| fwe | 0.125 | 0.400 | 0.420 | 0.660 | +0.260 |
| fwe | 0.0625 | 0.000 | 0.020 | 0.660 | +0.660 |
| qa_1 | 0.5 | 0.880 | 0.860 | 0.860 | -0.020 |
| qa_1 | 0.25 | 0.860 | 0.880 | 0.860 | +0.000 |
| qa_1 | 0.125 | 0.780 | 0.840 | 0.860 | +0.080 |
| qa_1 | 0.0625 | 0.720 | 0.760 | 0.840 | +0.120 |
| vt | 0.5 | 1.000 | 1.000 | 1.000 | +0.000 |
| vt | 0.25 | 1.000 | 0.980 | 1.000 | +0.000 |
| vt | 0.125 | 0.500 | 0.940 | 1.000 | +0.500 |
| vt | 0.0625 | 0.000 | 0.020 | 1.000 | +1.000 |

- **P3** (gate does no harm on dilution: gated ≥ plain − 1/N): worst gated−plain = -0.020 (= 1 input(s) at N=50; floor -0.020) → HOLDS
  (Large *positive* gated−plain on this model is the Qwen3 over-close protecting accuracy, not harm; P3 tests only the negative side.)

**Verdict (dynamickv_qwen3_4k):** allocation ≠ admission: adaptive allocation alone does not protect the capacity-bound task, and the gate adds the missing admission decision.

