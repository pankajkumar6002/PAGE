# E8 (W2): the headline matrix with a per-head Ada-KV plain arm

Both plain arms are recorded on the SAME inputs in the SAME run, so
every comparison below is paired. The only thing that varies is the
allocation of a fixed token budget across kv-heads.

## Delta (gated minus plain), by allocation

| cell | budgets | Delta shared | Delta Ada-KV | change | n |
|---|---|---:|---:|---:|---:|
| Qwen2.5-1.5B | all b<1.0 | +0.1206 | +0.0994 | -0.0212 | 3200 |
| Qwen2.5-1.5B | b<=0.25 | +0.1608 | +0.1608 | +0.0000 | 1200 |
| Qwen2.5-3B | all b<1.0 | +0.2597 | +0.1844 | -0.0753 | 3200 |
| Qwen2.5-3B | b<=0.25 | +0.3775 | +0.3475 | -0.0300 | 1200 |
| Qwen2.5-14B | all b<1.0 | +0.1412 | +0.1062 | -0.0350 | 3200 |
| Qwen2.5-14B | b<=0.25 | +0.2483 | +0.2367 | -0.0117 | 1200 |
| Mistral-7B | all b<1.0 | +0.1866 | +0.2116 | +0.0250 | 3200 |
| Mistral-7B | b<=0.25 | +0.2433 | +0.2417 | -0.0017 | 1200 |
| **mean** | all b<1.0 | **+0.1770** | **+0.1504** | **-0.0266** | |
| **mean** | b<=0.25 | **+0.2575** | **+0.2467** | **-0.0108** | |

## Per-task Delta at b <= 0.25

| cell | task | Delta shared | Delta Ada-KV | plain shared | plain Ada-KV |
|---|---|---:|---:|---:|---:|
| Qwen2.5-1.5B | niah_multikey_3 | +0.6433 | +0.6433 | 0.007 | 0.007 |
| Qwen2.5-1.5B | vt | +0.0000 | +0.0000 | 0.543 | 0.437 |
| Qwen2.5-1.5B | fwe | +0.0000 | +0.0000 | 0.147 | 0.133 |
| Qwen2.5-1.5B | qa_1 | +0.0000 | +0.0000 | 0.663 | 0.683 |
| Qwen2.5-3B | niah_multikey_3 | +0.9300 | +0.9233 | 0.000 | 0.007 |
| Qwen2.5-3B | vt | +0.0100 | +0.0167 | 0.627 | 0.550 |
| Qwen2.5-3B | fwe | +0.5700 | +0.4500 | 0.190 | 0.310 |
| Qwen2.5-3B | qa_1 | +0.0000 | +0.0000 | 0.767 | 0.750 |
| Qwen2.5-14B | niah_multikey_3 | +0.9933 | +0.9467 | 0.007 | 0.053 |
| Qwen2.5-14B | vt | +0.0000 | +0.0000 | 0.577 | 0.593 |
| Qwen2.5-14B | fwe | +0.0000 | +0.0000 | 0.470 | 0.450 |
| Qwen2.5-14B | qa_1 | +0.0000 | +0.0000 | 0.880 | 0.890 |
| Mistral-7B | niah_multikey_3 | +0.8900 | +0.8867 | 0.000 | 0.003 |
| Mistral-7B | vt | +0.0000 | +0.0000 | 0.660 | 0.543 |
| Mistral-7B | fwe | +0.0833 | +0.0800 | 0.420 | 0.373 |
| Mistral-7B | qa_1 | +0.0000 | +0.0000 | 0.797 | 0.810 |

## Pre-registered predictions

| id | prediction | outcome |
|---|---|---|
| P1 | Ada-KV delta <= shared delta in every cell | **VIOLATED** |
| P3 | mean Ada-KV delta at b<=0.25 >= +0.15 | HOLDS (measured +0.2467) |
| P4 | MK3 keeps the largest per-task delta | HOLDS |

## Reading

**The headline does not survive.** At b <= 0.25 the Ada-KV delta is
+0.2467, below the pre-registered floor of +0.15.
Per the falsification table this is a major finding: the matrix
result is substantially a single-mask artifact. Escalate before
writing.

**P1 violated at 4 (cell, budget) points** (aggregated over budgets it also fails):

| cell | b | Delta shared | Delta Ada-KV | excess |
|---|---:|---:|---:|---:|
| Mistral-7B | 0.0625 | +0.2500 | +0.2525 | +0.0025 |
| Mistral-7B | 0.625 | +0.1775 | +0.2025 | +0.0250 |
| Mistral-7B | 0.75 | +0.1075 | +0.1875 | +0.0800 |
| Mistral-7B | 0.875 | +0.0425 | +0.1475 | +0.1050 |

The pre-registration reads P1 failure as a code bug. That call needs
a magnitude check before it is accepted: seed SD measured in the
prior round was 0.004-0.011, so excesses inside that band are noise
rather than evidence of a defect. Audit the implementation against
the reference before either reporting or dismissing these numbers.

## Checks

- [FAIL] P1 per-budget direction (Ada-KV no stronger delta than shared)
- [PASS] at least one complete cell (4/4)
- [PASS] P3 survival at b<=0.25: +0.2467 vs floor +0.15
- [PASS] P4 task ordering preserved: True
- [INFO] P2 and P5 are not evaluated here: P2 needs the b=0.5 MK3
  breakdown and P5 is verified at run time by the runner's guards.
