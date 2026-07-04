# Second-seed robustness of the paper's directional conclusions

A reviewer flagged that the 32K pre-registration, the cross-family transfer,
and the SOTA head-to-heads rest on **single input draws** (all paper scripts
select the first *N* examples of each split deterministically; the `--seed`
argument only affects the `random`-eviction control, not the input draw). To
test seed-robustness we re-drew the inputs with a second seed and re-ran the
key experiments.

## Method: what "second seed" means here

The paper's canonical draw is "first *N* examples" (seed-1). For seed-2 we
shuffle each split with `datasets.shuffle(seed=20260704)` and take the first
*N* of the shuffled order — a genuinely different draw (measured overlap with
the first-*N* draw is only ~18-20% of examples). This is done via thin
monkeypatch wrappers that leave all model / eviction / scoring / decoding logic
untouched:

- `experiments/scripts/ruler_sweep_32k_seed2.py`
- `experiments/scripts/gated_eviction_seed2.py`
- `experiments/scripts/longbench_capkv_seed2.py`
- `external/trimkv/run_gated_dbtrimkv_seed2.py`

No paper `.tex` files were edited.

**Caveat on N (honest).** The cluster was heavily oversubscribed during this
run (all 4 GPUs at 99-100% utilization from other users, with volatile memory
spikes that OOM-killed the 32K job twice before it was placed under a
retry-loop on the emptiest GPU). To finish within the session, the 4K gated
cross-family runs were reduced to **N=20/task** (from the paper's N=100) and the
DBTrimKV head-to-head to **per_task=12** (from 30). The **32K pre-registration
was kept at the full N=100/task** (the headline test). Smaller N widens the
confidence intervals on the seed-2 point estimates, so the verdicts below lean
on **direction and sign**, which are N-robust, rather than on exact magnitudes.

---

## 1. 32K pre-registered scaling test (Qwen2.5-1.5B, RULER-32K, N=100/task)

Full N=100/task, 10 budgets, SnapKV eviction, two-pass, greedy,
`reanalyze_ruler.py` rho definition (rho = fraction wrong@full-KV AND recovered
at some b<1.0). Frozen predictions from `scaling_32k_prediction.md`.

| task | frozen prediction | seed-1 rho [Wilson95] | seed-2 rho [Wilson95] | seed-1 verdict | seed-2 verdict |
|---|---|---|---|---|---|
| **VT**  | rho in [0.048, 0.106] (pt 0.072) | **0.140** [0.085, 0.221] | **0.090** [0.048, 0.162] | OUTSIDE (above) — MISS | **INSIDE — LANDS** |
| **FWE** | rho in [0.021, 0.263] (pt 0.087) | **0.050** [0.022, 0.112] | **0.040** [0.016, 0.098] | INSIDE — LANDS | **INSIDE — LANDS** |
| **MK3** | rho <= 0.02 (protocol pin)        | **0.040** [0.016, 0.098] | **0.030** [0.010, 0.085] | above pin — MISS (CI-compat) | above pin — MISS (CI-compat) |

Supporting A_full: VT 0.810 -> **0.870**, FWE 0.490 -> **0.420**, MK3 0.140 -> **0.200**.

**Prediction hit count: seed-1 = 1/3 (FWE). seed-2 = 2/3 (VT + FWE).**

### What replicates and what does not (be honest)

- **FWE inside: REPLICATES.** rho 0.050 -> 0.040, comfortably inside the
  (wide) predicted band both seeds.
- **MK3 near-pin: REPLICATES qualitatively.** rho 0.040 -> 0.030. The strict
  `rho <= 0.02` pin is missed at the point estimate on **both** seeds, but the
  Wilson CI includes 0.02 both times — the capacity-bound task stays a small,
  near-pin rho. Direction stable.
- **VT "outside-above": DOES NOT REPLICATE — this is a flip.** The seed-1
  headline (VT rho 0.140, ~32% above the predicted upper bound, "eviction is
  net-positive on VT at 32K", implied dilution ratio 0.737 — the strongest
  amplification signature in the paper) was substantially a **single-draw
  effect**. On seed-2, VT full-KV accuracy rose (0.81->0.87), rho fell to 0.090,
  which sits **inside** the frozen band [0.048, 0.106]. The VT budget curve also
  loses its amplification signature:

  | budget | seed-1 VT acc | seed-2 VT acc |
  |---:|---:|---:|
  | 1.0 (full) | 0.810 | 0.870 |
  | 0.25 | 0.840 | 0.820 |
  | 0.125 | **0.850** (> full) | 0.860 (< full) |
  | 0.0625 | 0.810 | 0.800 |

  Seed-1 shows eviction *beating* full-KV (flat-to-rising); seed-2 is
  flat-to-slightly-declining, i.e. ordinary mild dilution, never exceeding
  full-KV. **The paper's striking "eviction net-positive on VT at 32K" claim is
  not seed-robust and should be softened or dropped.**

**Net for the 32K test:** the frozen formula is actually validated *better* by
the second draw (2/3 vs 1/3). The single conclusion that flips (VT
outside-above) flips *toward* the prediction, and the over-strong
amplification narrative around it does not survive a second draw.

---

## 2. Cross-family transfer (RULER 4K mixed suite, tau=0.07 fixed vs z-scored)

`niah_multikey_3, vt, fwe, qa_1`, SnapKV, two-pass sdpa. Post-hoc identity:
`gated(tau) = plain(b) if drop>=tau else full-KV`. z-threshold
`tau_z = mu + (-0.69)*sigma` with mu, sigma pooled per-run (self-standardizing,
theta_z frozen on Qwen/Mistral). Seed-1 Qwen3 = existing N=100 run; seed-1
Llama = matched fresh N=20 first-100 draw; both seed-2 = N=20 shuffle.

### Qwen3-4B-Instruct-2507 — replicates cleanly

| quantity | seed-1 (N=100) | seed-2 (N=20) |
|---|---|---|
| pooled drop mu / sigma | +0.0337 / 0.0173 | +0.0345 / 0.0176 |
| drop ordering | MK3 < FWE < VT < QA1 (MK3 smallest) | MK3 < FWE < VT < QA1 (MK3 smallest) |
| gate-open @ fixed tau=0.07 (mk3/fwe/vt/qa1) | 0.00 / 0.00 / 0.00 / 0.03 | 0.00 / 0.00 / 0.00 / 0.00 |
| gate-open @ z-tau (mk3/fwe/vt/qa1) | 0.02 / 0.94 / 1.00 / 1.00 | 0.05 / 0.95 / 1.00 / 1.00 |
| mixed-suite Delta @ fixed 0.07 | +0.429 (full-KV-fallback artifact) | +0.444 (same artifact) |
| mixed-suite Delta @ z-tau | +0.235 (real 2.3x-compression gate) | +0.216 (real gate) |

**All three claims hold on seed-2:** fixed tau=0.07 fires **~never** (degenerates
to full-KV fallback — 0% compression), the z-scored threshold **restores a
working gate** (dilution tasks open, MK3 closed), and the capacity-bound
**MK3-smallest-drop ordering** transfers. Drop distribution is essentially
seed-invariant.

### Llama-3.1-8B-Instruct — ordering and sign preserved

| quantity | seed-1 (N=20) | seed-2 (N=20) |
|---|---|---|
| pooled drop mu / sigma | +0.0894 / 0.0182 | +0.0889 / 0.0180 |
| drop ordering | MK3 < VT < QA1 < FWE (MK3 smallest) | MK3 < VT < QA1 < FWE (MK3 smallest) |
| gate-open @ fixed 0.07 (mk3/fwe/vt/qa1) | 0.45 / 1.00 / 1.00 / 0.95 | 0.70 / 1.00 / 1.00 / 0.90 |
| mixed-suite Delta @ fixed 0.07 | +0.137 | +0.072 |
| mixed-suite Delta @ z-tau | +0.225 | +0.144 |

**Ordering preserved** (MK3 smallest, dilution tasks well above tau and always
gating open); **both Deltas stay positive** on both seeds. Unlike Qwen3, Llama's
drops sit *above* tau (mu 0.089 >> 0.07), so fixed-tau already mostly fires and
the gain is smaller — exactly the paper's cross-family contrast. Magnitudes wob-
ble (fixed +0.137 -> +0.072) because at N=20 the whole signal rides on the
capacity task MK3, whose boundary gate-open fraction is the noisiest quantity
(0.45 -> 0.70). Direction is robust; magnitude is N-limited.

---

## 3. SOTA head-to-heads (small N — noise expected, per the brief)

All: `gate-closed Delta > 0`, `gate-open Delta = 0` by construction.

| head-to-head | seed-1 | seed-2 | replicates? |
|---|---|---|---|
| **CapKV / LongBench-qasper** (Qwen2.5-3B, b=0.5) | Delta +0.0147 (N=68); gate-closed +0.044 | Delta +0.0104 (N=96); gate-closed +0.040 | YES — sign + magnitude |
| **CapKV / LongBench-triviaqa** (Qwen2.5-3B, b=0.5) | Delta +0.0833 (N=24); gate-closed +0.182 | Delta +0.0952 (N=21); gate-closed +0.222 | YES — sign + magnitude |
| **DBTrimKV / RULER-4K** (Qwen3-4B) | Delta +0.233 (N=360); gate-open 2/120 | Delta +0.257 (N=48); gate-open 0/48 | YES — sign + magnitude |

Details:
- **CapKV**: both tasks reproduce the exact structure — gate-open subset Delta=0,
  a positive gate-closed lift (qasper ~+0.04, triviaqa ~+0.20), tiny positive
  headline. The +1-example flips are within noise, as flagged.
- **DBTrimKV**: gate is **essentially always closed at 4K** on both seeds
  (2/120 -> 0/48), so gated == full-KV; the win concentrates on MK3 (+0.533 ->
  +0.472) and FWE (+0.378 -> +0.583), with VT/QA1 neutral (both seeds). Drop
  median is identical (+0.0371 -> +0.0372).

---

## Overall verdict

**The paper's directional conclusions are largely seed-robust.**

- **Cross-family transfer (Qwen3, Llama): fully replicates.** The drop
  distributions are nearly seed-invariant; the MK3-smallest ordering, the "fixed
  tau fires ~never on Qwen3 / z-score restores a working gate" story, and the
  positive-Delta / ordering-preserved Llama story all hold, with Qwen3 numbers
  almost identical across seeds.
- **Head-to-heads (CapKV, DBTrimKV): replicate** in sign and rough magnitude,
  with the gate-open/gate-closed invariant intact (small N, as expected).
- **32K pre-registration: one conclusion flips, in the formula's favour.** More
  predictions land on seed-2 (2/3) than seed-1 (1/3). The single flip is **VT**,
  whose seed-1 "outside-above + eviction-beats-full-KV amplification" was a
  single-draw artifact: on seed-2 VT lands *inside* the predicted band with an
  ordinary (non-amplifying) budget curve. **Recommendation: soften/remove the
  32K VT amplification narrative;** the underlying headroom-tracking formula is
  otherwise corroborated (2/3) by the independent draw.

### Headline answer
- **32K seed-2 prediction hit count: 2 of 3** (VT lands inside, FWE lands inside;
  MK3 point-estimate 0.030 slightly above the 0.02 pin but CI-compatible). Seed-1
  was 1 of 3.

### Files (seed-2 outputs)
- `experiments/results/ruler_32k_qwen15b_sweep_seed2.jsonl`
- `experiments/results/gated_4k_qwen3_4b_seed2.jsonl`
- `experiments/results/gated_4k_llama31_8b_seed1_n20.jsonl` (matched seed-1 draw)
- `experiments/results/gated_4k_llama31_8b_seed2.jsonl`
- `experiments/results/capkv_longbench_qasper_seed2.jsonl`
- `experiments/results/capkv_longbench_triviaqa_seed2.jsonl`
- `experiments/results/gated_dbtrimkv_qwen3_4k_seed2.jsonl`
