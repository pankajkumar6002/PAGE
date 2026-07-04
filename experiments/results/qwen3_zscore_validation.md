# Qwen3-4B z-score transfer validation (third architecture family)

Setup: Qwen3-4B-Instruct-2507, RULER 4K mixed suite (niah_multikey_3, vt, fwe,
qa_1), N=100/task, budgets {1.0, 0.5, 0.25, 0.125, 0.0625}, score_policy=snapkv.
The run was split across two files (an ETA-waiter died mid-run and the sweep
resumed into a scratchpad file); merged and deduped to
`gated_4k_qwen3_4b_merged.jsonl` (2000 records = 4 tasks x 100 ids x 5 budgets,
verified complete coverage). Post-hoc identity used throughout: gated(b) =
correct_plain(b) if drop >= tau else full-KV outcome (b=1.0 correct_plain).

## Why Qwen3 is the sharpest transfer test

Qwen3-4B's head-agreement drop lives on a completely different scale from
Qwen2.5: pooled over all 400 inputs, mu = 0.0337, sd = 0.0173. The paper's
fixed tau = 0.07 sits at z = +1.98 on Qwen3 -- above almost every input -- so
the gate essentially never fires. This is the same failure the DBTrimKV
head-to-head flagged (gate open on 1.7% of Qwen3 records). Qwen3 is therefore
the cleanest case for the z-scored variant: fixed tau fails completely, so any
working gate must come from standardization.

## Ordering check (transfers)

Per-task mean drop, smallest first:

| task | mean drop |
|---|---:|
| niah_multikey_3 | +0.010 |
| fwe | +0.030 |
| vt  | +0.040 |
| qa_1 | +0.054 |

NIAH-MK3 is the smallest drop -- the capacity-bound ordering transfers to
Qwen3 (unlike the Llama family, where niah_multivalue displaces it).

## Fixed tau vs z-derived tau

z-derived threshold: tau_qwen3 = mu + (-0.69)*sd = **0.0217** (theta_z = -0.69
fit on Qwen/Mistral only, unchanged).

| threshold | plain | gated | Delta | mean kept-KV | gate-open (mk3 / fwe / vt / qa_1) |
|---|---:|---:|---:|---:|---|
| fixed tau=0.07 | 0.456 | 0.885 | +0.429 | **0.994** | 0.00 / 0.00 / 0.00 / 0.03 |
| z-tau=0.0217 | 0.456 | 0.691 | +0.235 | **0.433** | 0.02 / 0.94 / 1.00 / 1.00 |

- **Fixed tau=0.07 is a full-KV-fallback artifact**: it keeps 99.4% of the
  cache (0.6% compression) and the +0.429 is not a compression result at all.
- **z-tau=0.0217 is a real gate**: MK3 closed (2% open), dilution tasks open
  (94-100%), 43% mean kept-KV (2.3x compression), +23.5pp over plain.

Per-task at z-tau: MK3 plain 0.048 -> gated 0.983 (+93.5pp, the gate closes and
recovers full-KV); fwe/vt/qa_1 approximately neutral (+0.5/0/0 pp) -- the gate
opens and gated == plain by construction, as intended on dilution-prone tasks.

Oracle (per-input max of evict/full, pooled b<1) = 0.887; the z-tau gate
recovers (0.691-0.456)/(0.887-0.456) = **54.5%** of available headroom.

## Verdict

Qwen3-4B is a third held-out architecture family (a different model generation
from the Qwen2.5 calibration set) on which:
1. the capacity-bound drop ordering (MK3 smallest) transfers;
2. fixed tau=0.07 fails completely -- it degenerates to full-KV fallback with
   0.6% compression -- precisely because Qwen3's drop scale differs;
3. the z-scored threshold (theta_z=-0.69, calibrated on Qwen/Mistral, applied
   via an unlabeled per-model pilot) restores a correct partition and real
   2.3x compression at +23.5pp.

This is a stronger demonstration than Yi/Llama that the drop is an
architecture-invariant *ordering* whose scale must be standardized per model;
here fixed-tau does not merely underperform, it does nothing.
