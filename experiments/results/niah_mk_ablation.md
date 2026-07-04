# Near-tie distractor ablation: niah_multikey_{1,2,3}

Qwen2.5-1.5B-Instruct, RULER 4K, N=50 per task. tau=0.07. Budgets: 1, 0.5, 0.25, 0.125, 0.0625

## Per-task mean D and recovery rates

| task | N | mean D | rho_plain | rho_gated | gate-open frac |
|---|---:|---:|---:|---:|---:|
| niah_multikey_1 | 50 | +0.1550 | 0.040 | 0.040 | 1.000 |
| niah_multikey_2 | 50 | +0.1219 | 0.000 | 0.000 | 1.000 |
| niah_multikey_3 | 50 | +0.0448 | 0.000 | 0.000 | 0.000 |

## Per-task per-budget accuracy

### niah_multikey_1  (N=50)

| budget | plain acc | gated acc | delta |
|---:|---:|---:|---:|
| 1 | 0.960 | 0.960 | +0.000 |
| 0.5 | 0.960 | 0.960 | +0.000 |
| 0.25 | 0.980 | 0.980 | +0.000 |
| 0.125 | 0.960 | 0.960 | +0.000 |
| 0.0625 | 0.360 | 0.360 | +0.000 |

### niah_multikey_2  (N=50)

| budget | plain acc | gated acc | delta |
|---:|---:|---:|---:|
| 1 | 0.800 | 0.800 | +0.000 |
| 0.5 | 0.120 | 0.120 | +0.000 |
| 0.25 | 0.020 | 0.020 | +0.000 |
| 0.125 | 0.020 | 0.020 | +0.000 |
| 0.0625 | 0.000 | 0.000 | +0.000 |

### niah_multikey_3  (N=50)

| budget | plain acc | gated acc | delta |
|---:|---:|---:|---:|
| 1 | 0.660 | 0.660 | +0.000 |
| 0.5 | 0.100 | 0.660 | +0.560 |
| 0.25 | 0.020 | 0.660 | +0.640 |
| 0.125 | 0.000 | 0.660 | +0.660 |
| 0.0625 | 0.000 | 0.660 | +0.660 |

## Per-task D (head-agreement drop) detail

| task | mean D | min D | median D | max D | n |
|---|---:|---:|---:|---:|---:|
| niah_multikey_1 | +0.1550 | +0.1132 | +0.1591 | +0.1996 | 50 |
| niah_multikey_2 | +0.1219 | +0.0883 | +0.1205 | +0.1597 | 50 |
| niah_multikey_3 | +0.0448 | +0.0272 | +0.0445 | +0.0623 | 50 |

## Gradient verdict

- D order: niah_multikey_1(+0.1550) > niah_multikey_2(+0.1219) > niah_multikey_3(+0.0448)
- rho_plain order: niah_multikey_1(0.040) > niah_multikey_2(0.000) > niah_multikey_3(0.000)
- gate-open frac order: niah_multikey_1(1.000) > niah_multikey_2(1.000) > niah_multikey_3(0.000)

- Predicted gradient: niah_mk_1 > niah_mk_2 > niah_mk_3 on D, rho_plain, gate-open frac.
  - D gradient holds: True
  - rho_plain gradient holds (weakly): True
  - gate-open gradient holds (weakly): True

## Interpretation

The **D gradient is clean and monotonic**: niah_mk_1 (0 near-tie distractors) D=+0.1550, niah_mk_2 (1 near-tie distractor) D=+0.1219, niah_mk_3 (2 near-tie distractors) D=+0.0448. More near-tie distractors -> smaller head-agreement drop. This is a direct mechanism-validation result: the per-input agreement signal tracks the number of competing keys.

At tau=0.07 the partition is binary: niah_mk_1 and niah_mk_2 are both fully "above the line" (gate-open 100%), niah_mk_3 is fully "below" (gate-open 0%). The min D in niah_mk_3 (+0.0272) and the max D in niah_mk_3 (+0.0623) both sit below tau=0.07; niah_mk_2's min (+0.0883) sits above. So the predictor cleanly separates 2-distractor from 0- and 1-distractor cases.

rho_plain at this slice is small in absolute terms because (a) on niah_mk_1 the model holds 96-98 percent up to b=0.125 (little room for "full-wrong, lower-correct"), and (b) on niah_mk_2 the plain SnapKV collapses from 80 percent at full-KV to 12 percent at b=0.5 and 0 percent at b=0.0625 -- so plain never "recovers" niah_mk_2 inputs. niah_mk_2 is in fact **also capacity-bound** at low budgets despite its larger D: the predictor sees dilution-prone behavior in head agreement but the retrieval task still needs the keys. The gated policy correctly applies eviction on niah_mk_1 (where plain accuracy is preserved) and skips eviction on niah_mk_3 (gated acc = 0.660 across all budgets vs plain collapse to 0). On niah_mk_2 the gate stays open and tracks plain accuracy down with budget -- this is the "false-positive" case for the gate at tau=0.07 and is the right place to look for predictor refinement.

Caveat: niah_mk_2 result suggests "near-tie count" alone does not pick out the capacity-bound regime. The partition predictor catches niah_mk_3 (heavy near-tie load) but not niah_mk_2 (single near-tie distractor at this model scale / context length). The gradient still holds on D, which is the headline.
