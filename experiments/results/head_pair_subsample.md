# P3-3: head-pair subsampling

Qwen2.5-1.5B, RULER 4K, $H = 12$ heads, 66 pairs, $L = 28$ layers, 120 inputs.

| pairs kept | count | mean $D$ MK3 | min mean $D$ dil. | margin | ordering |
|---|---:|---:|---:|---:|---|
| 100% | 66 | 0.0413 | 0.1131 | +0.0717 | 1/1 draws |
| 50% | 33 | 0.0404 | 0.1109 | +0.0705 | 20/20 draws |
| 25% | 16 | 0.0456 | 0.1179 | +0.0724 | 20/20 draws |
| 10% | 7 | 0.0280 | 0.1035 | +0.0754 | 20/20 draws |
| 5% | 3 | 0.0541 | 0.1206 | +0.0665 | 20/20 draws |
| 2% | 1 | 0.0529 | 0.1264 | +0.0735 | 20/20 draws |

The ordering survives on every draw down to **1 of 66 pairs** (2%), where the margin is +0.0735 against +0.0717 at full cost.

Reading: the $H^2$ term can be cut to about 2% without losing the separation the gate thresholds. On a 40-head model that is 15 pairs instead of 780, which is the difference the deployment objection turns on.
