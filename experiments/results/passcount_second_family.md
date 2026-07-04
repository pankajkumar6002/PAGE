# passage_count as a second capacity-bound family (Qwen2.5-14B)

Question: is LongBench passage_count a second, non-RULER task the gate classifies
capacity-bound a priori, once run on a model above floor accuracy? (On Qwen2.5-1.5B
it was gate-predicted capacity-bound but untestable: full-KV accuracy 0.043 ~ chance.)

Setup: Qwen2.5-14B-Instruct, LongBench passage_count, budgets {1.0,0.5,0.25,0.125,0.0625},
N=95, two-pass. Raw: longbench_passcount_qwen14b.jsonl.

## Result

| budget | accuracy |
|---|---:|
| 1.0 (full) | 0.158 |
| 0.5 | 0.147 |
| 0.25 | 0.158 |
| 0.125 | 0.137 |
| 0.0625 | 0.126 |

- A_full = 0.158 (above the 1.5B floor of 0.043, but still a hard task).
- rho = 0.011 (1/95), well below the 0.05 dilution-prone threshold.
- mean head-agreement drop D = -0.013; gate CLOSED (D < tau=0.07) on 100% of inputs.

## Verdict (case a, muted)

The gate a-priori classifies passage_count capacity-bound: mean D is negative
(the same signature as NIAH-MK3), so the gate closes on every input and keeps
the full cache. Eviction does not help (rho = 0.01), consistent with
capacity-bound. This is a second, non-RULER task on which the a-priori gate
prediction is correct, which strengthens the predictor claim beyond RULER.

Caveat (honest): because full-KV accuracy is only 0.158 even at 14B, the
"eviction destroys accuracy" signature is muted (0.158 -> 0.126 at b=0.0625, a
gentle decline, not MK3's 0.99 -> 0.00 collapse). passage_count is a hard task
with a low ceiling, so it confirms the gate's a-priori classification and the
no-help property, but it is not a dramatic accuracy-collapse showcase. The
strong showcase remains MK3.

Paper sentence: "On a second, non-RULER task (LongBench passage_count,
Qwen2.5-14B), the gate again classifies capacity-bound a priori (mean D negative,
closed on 100% of inputs) and eviction does not help (rho = 0.01); the task's low
ceiling (full-KV 0.16) makes it a confirmation of the a-priori prediction rather
than an accuracy-collapse showcase."
