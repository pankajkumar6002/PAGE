
## Scorer-based gates at a matched open rate

- Qwen2.5-1.5B: joined 400 inputs, T mismatches 0, closures at tau 100, |D run - logged| max 0.0063, correlation 0.9997.
- Qwen2.5-3B: joined 400 inputs, T mismatches 0, closures at tau 204, |D run - logged| max 0.0030, correlation 0.9998.
- Mistral-7B: joined 400 inputs, T mismatches 0, closures at tau 113, |D run - logged| max 0.0003, correlation 1.0000.
- Qwen2.5-14B: joined 200 inputs, T mismatches 0, closures at tau 50, |D run - logged| max 0.0195, correlation 0.9976.

### Table 41: Gate gain over plain SnapKV at D's closed count

| model | signal | orientation (fixed on Qwen2.5-1.5B) | gain | MK3 closed | gain, hindsight orientation | D minus signal | 95% CI | AUC for MK3 |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | a_early | low closes | +0.121 | 100 | +0.121 | +0.000 | [+0.000, +0.000] | 1.000 |
| Qwen2.5-1.5B | entropy_drop | high closes | +0.121 | 100 | +0.121 | +0.000 | [+0.000, +0.000] | 1.000 |
| Qwen2.5-1.5B | top32_share | low closes | +0.057 | 20 | +0.057 | +0.064 | [+0.042, +0.083] | 0.711 |
| Qwen2.5-1.5B | snapkv@0.125 | low closes | +0.016 | 2 | +0.030 | +0.105 | [+0.085, +0.129] | 0.667 |
| Qwen2.5-1.5B | h2o@0.125 | low closes | +0.011 | 0 | +0.030 | +0.110 | [+0.088, +0.131] | 0.656 |
| Qwen2.5-1.5B | obcache_v@0.125 | low closes | +0.020 | 8 | +0.020 | +0.101 | [+0.081, +0.122] | 0.674 |
| Qwen2.5-1.5B | obcache_k@0.125 | low closes | +0.016 | 3 | +0.030 | +0.105 | [+0.084, +0.126] | 0.668 |
| Qwen2.5-1.5B | obcache_vk@0.125 | low closes | +0.019 | 6 | +0.030 | +0.102 | [+0.081, +0.126] | 0.671 |
| Qwen2.5-1.5B | act_change@0.125 | high closes | +0.043 | 29 | +0.043 | +0.078 | [+0.057, +0.096] | 0.585 |
| Qwen2.5-1.5B | kl@0.125 | low closes | +0.060 | 48 | +0.060 | +0.061 | [+0.042, +0.076] | 0.732 |
| Qwen2.5-1.5B | top1_same@0.125 | high closes | +0.043 | 26 | +0.043 | +0.078 | [+0.063, +0.093] | 0.520 |
| Qwen2.5-1.5B | D (logged) | low closes | +0.121 | 100 | +0.121 |  |  | 1.000 |
| Qwen2.5-1.5B | random gate |  | +0.042 |  |  | +0.079 |  |  |
| Qwen2.5-3B | a_early | low closes | +0.220 | 100 | +0.220 | +0.039 | [+0.027, +0.053] | 1.000 |
| Qwen2.5-3B | entropy_drop | high closes | +0.229 | 100 | +0.229 | +0.031 | [+0.020, +0.043] | 1.000 |
| Qwen2.5-3B | top32_share | low closes | +0.220 | 100 | +0.220 | +0.039 | [+0.027, +0.054] | 0.668 |
| Qwen2.5-3B | snapkv@0.125 | low closes | +0.189 | 93 | +0.189 | +0.070 | [+0.054, +0.083] | 0.661 |
| Qwen2.5-3B | h2o@0.125 | low closes | +0.090 | 4 | +0.222 | +0.170 | [+0.154, +0.184] | 0.667 |
| Qwen2.5-3B | obcache_v@0.125 | low closes | +0.182 | 90 | +0.182 | +0.078 | [+0.060, +0.094] | 0.650 |
| Qwen2.5-3B | obcache_k@0.125 | low closes | +0.194 | 100 | +0.194 | +0.065 | [+0.051, +0.080] | 0.761 |
| Qwen2.5-3B | obcache_vk@0.125 | low closes | +0.194 | 100 | +0.194 | +0.065 | [+0.050, +0.080] | 0.741 |
| Qwen2.5-3B | act_change@0.125 | high closes | +0.243 | 85 | +0.243 | +0.016 | [+0.009, +0.028] | 0.710 |
| Qwen2.5-3B | kl@0.125 | low closes | +0.202 | 86 | +0.202 | +0.057 | [+0.042, +0.075] | 0.873 |
| Qwen2.5-3B | top1_same@0.125 | high closes | +0.158 | 54 | +0.158 | +0.101 | [+0.092, +0.111] | 0.535 |
| Qwen2.5-3B | D (logged) | low closes | +0.259 | 100 | +0.259 |  |  | 1.000 |
| Qwen2.5-3B | random gate |  | +0.154 |  |  | +0.105 |  |  |
| Mistral-7B | a_early | low closes | +0.198 | 100 | +0.198 | -0.011 | [-0.022, -0.000] | 1.000 |
| Mistral-7B | entropy_drop | high closes | +0.137 | 64 | +0.137 | +0.050 | [+0.034, +0.067] | 0.876 |
| Mistral-7B | top32_share | low closes | +0.066 | 21 | +0.066 | +0.121 | [+0.104, +0.138] | 0.621 |
| Mistral-7B | snapkv@0.125 | low closes | +0.027 | 13 | +0.039 | +0.160 | [+0.148, +0.172] | 0.624 |
| Mistral-7B | h2o@0.125 | low closes | +0.010 | 0 | +0.058 | +0.177 | [+0.162, +0.192] | 0.667 |
| Mistral-7B | obcache_v@0.125 | low closes | +0.028 | 13 | +0.040 | +0.159 | [+0.147, +0.173] | 0.667 |
| Mistral-7B | obcache_k@0.125 | low closes | +0.026 | 13 | +0.038 | +0.161 | [+0.149, +0.173] | 0.666 |
| Mistral-7B | obcache_vk@0.125 | low closes | +0.028 | 13 | +0.038 | +0.159 | [+0.148, +0.173] | 0.667 |
| Mistral-7B | act_change@0.125 | high closes | +0.166 | 84 | +0.166 | +0.021 | [+0.005, +0.037] | 0.932 |
| Mistral-7B | kl@0.125 | low closes | +0.018 | 0 | +0.194 | +0.169 | [+0.154, +0.182] | 0.985 |
| Mistral-7B | top1_same@0.125 | high closes | +0.075 | 25 | +0.091 | +0.112 | [+0.100, +0.124] | 0.572 |
| Mistral-7B | D (logged) | low closes | +0.187 | 90 | +0.187 |  |  | 0.977 |
| Mistral-7B | random gate |  | +0.080 |  |  | +0.107 |  |  |
| Qwen2.5-14B | a_early | low closes | +0.143 | 50 | +0.143 | +0.000 | [+0.000, +0.000] | 1.000 |
| Qwen2.5-14B | entropy_drop | high closes | +0.143 | 50 | +0.143 | +0.000 | [+0.000, +0.000] | 1.000 |
| Qwen2.5-14B | top32_share | low closes | +0.039 | 0 | +0.053 | +0.104 | [+0.092, +0.117] | 0.660 |
| Qwen2.5-14B | snapkv@0.125 | low closes | +0.001 | 0 | +0.091 | +0.143 | [+0.129, +0.160] | 0.821 |
| Qwen2.5-14B | h2o@0.125 | low closes | +0.035 | 0 | +0.039 | +0.108 | [+0.086, +0.130] | 0.575 |
| Qwen2.5-14B | obcache_v@0.125 | low closes | -0.004 | 0 | +0.056 | +0.147 | [+0.134, +0.163] | 0.545 |
| Qwen2.5-14B | obcache_k@0.125 | low closes | +0.014 | 7 | +0.049 | +0.129 | [+0.109, +0.146] | 0.703 |
| Qwen2.5-14B | obcache_vk@0.125 | low closes | +0.014 | 6 | +0.054 | +0.129 | [+0.112, +0.148] | 0.696 |
| Qwen2.5-14B | act_change@0.125 | high closes | +0.074 | 27 | +0.074 | +0.069 | [+0.053, +0.089] | 0.828 |
| Qwen2.5-14B | kl@0.125 | low closes | +0.109 | 36 | +0.109 | +0.034 | [+0.017, +0.051] | 0.887 |
| Qwen2.5-14B | top1_same@0.125 | high closes | +0.058 | 12 | +0.058 | +0.085 | [+0.076, +0.095] | 0.500 |
| Qwen2.5-14B | D (logged) | low closes | +0.143 | 50 | +0.143 |  |  | 1.000 |
| Qwen2.5-14B | random gate |  | +0.058 |  |  | +0.085 |  |  |

### Pre-registered decision rule, three main models

| signal | models with D above | models with signal above | verdict |
|---|---|---|---|
| a_early | 1/3 | 1/3 | no clear advantage |
| entropy_drop | 2/3 | 0/3 | D has an advantage |
| top32_share | 3/3 | 0/3 | D has an advantage |
| snapkv@0.125 | 3/3 | 0/3 | D has an advantage |
| h2o@0.125 | 3/3 | 0/3 | D has an advantage |
| obcache_v@0.125 | 3/3 | 0/3 | D has an advantage |
| obcache_k@0.125 | 3/3 | 0/3 | D has an advantage |
| obcache_vk@0.125 | 3/3 | 0/3 | D has an advantage |
| act_change@0.125 | 3/3 | 0/3 | D has an advantage |
| kl@0.125 | 3/3 | 0/3 | D has an advantage |
| top1_same@0.125 | 3/3 | 0/3 | D has an advantage |

### Against per-input harm, in the orientation that closes MK3

| model | b | signal | harmed | pooled AUC | within-task AUC |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | 1/16 | D (logged) | 174 | 0.869 | 0.403 |
| Qwen2.5-1.5B | 1/16 | a_early | 174 | 0.925 | 0.475 |
| Qwen2.5-1.5B | 1/16 | top32_share | 174 | 0.862 | 0.518 |
| Qwen2.5-1.5B | 1/16 | snapkv@0.0625 | 174 | 0.136 | 0.548 |
| Qwen2.5-1.5B | 1/16 | h2o@0.0625 | 174 | 0.113 | 0.383 |
| Qwen2.5-1.5B | 1/16 | obcache_v@0.0625 | 174 | 0.179 | 0.471 |
| Qwen2.5-1.5B | 1/16 | obcache_k@0.0625 | 174 | 0.131 | 0.471 |
| Qwen2.5-1.5B | 1/16 | obcache_vk@0.0625 | 174 | 0.139 | 0.487 |
| Qwen2.5-1.5B | 1/16 | act_change@0.0625 | 174 | 0.738 | 0.639 |
| Qwen2.5-1.5B | 1/16 | kl@0.0625 | 174 | 0.294 | 0.422 |
| Qwen2.5-1.5B | 1/4 | D (logged) | 80 | 0.918 | 0.418 |
| Qwen2.5-1.5B | 1/4 | a_early | 80 | 0.878 | 0.476 |
| Qwen2.5-1.5B | 1/4 | top32_share | 80 | 0.534 | 0.531 |
| Qwen2.5-1.5B | 1/4 | snapkv@0.25 | 80 | 0.585 | 0.679 |
| Qwen2.5-1.5B | 1/4 | h2o@0.25 | 80 | 0.564 | 0.524 |
| Qwen2.5-1.5B | 1/4 | obcache_v@0.25 | 80 | 0.604 | 0.749 |
| Qwen2.5-1.5B | 1/4 | obcache_k@0.25 | 80 | 0.593 | 0.707 |
| Qwen2.5-1.5B | 1/4 | obcache_vk@0.25 | 80 | 0.597 | 0.713 |
| Qwen2.5-1.5B | 1/4 | act_change@0.25 | 80 | 0.643 | 0.651 |
| Qwen2.5-1.5B | 1/4 | kl@0.25 | 80 | 0.584 | 0.372 |
| Qwen2.5-3B | 1/16 | D (logged) | 280 | 0.934 | 0.507 |
| Qwen2.5-3B | 1/16 | a_early | 280 | 0.945 | 0.489 |
| Qwen2.5-3B | 1/16 | top32_share | 280 | 0.719 | 0.536 |
| Qwen2.5-3B | 1/16 | snapkv@0.0625 | 280 | 0.076 | 0.532 |
| Qwen2.5-3B | 1/16 | h2o@0.0625 | 280 | 0.040 | 0.391 |
| Qwen2.5-3B | 1/16 | obcache_v@0.0625 | 280 | 0.083 | 0.546 |
| Qwen2.5-3B | 1/16 | obcache_k@0.0625 | 280 | 0.141 | 0.567 |
| Qwen2.5-3B | 1/16 | obcache_vk@0.0625 | 280 | 0.134 | 0.570 |
| Qwen2.5-3B | 1/16 | act_change@0.0625 | 280 | 0.643 | 0.548 |
| Qwen2.5-3B | 1/16 | kl@0.0625 | 280 | 0.528 | 0.534 |
| Qwen2.5-3B | 1/4 | D (logged) | 141 | 0.946 | 0.363 |
| Qwen2.5-3B | 1/4 | a_early | 141 | 0.783 | 0.259 |
| Qwen2.5-3B | 1/4 | top32_share | 141 | 0.374 | 0.338 |
| Qwen2.5-3B | 1/4 | snapkv@0.25 | 141 | 0.580 | 0.307 |
| Qwen2.5-3B | 1/4 | h2o@0.25 | 141 | 0.490 | 0.238 |
| Qwen2.5-3B | 1/4 | obcache_v@0.25 | 141 | 0.448 | 0.317 |
| Qwen2.5-3B | 1/4 | obcache_k@0.25 | 141 | 0.548 | 0.288 |
| Qwen2.5-3B | 1/4 | obcache_vk@0.25 | 141 | 0.525 | 0.295 |
| Qwen2.5-3B | 1/4 | act_change@0.25 | 141 | 0.839 | 0.487 |
| Qwen2.5-3B | 1/4 | kl@0.25 | 141 | 0.707 | 0.608 |
| Mistral-7B | 1/16 | D (logged) | 263 | 0.686 | 0.467 |
| Mistral-7B | 1/16 | a_early | 263 | 0.783 | 0.539 |
| Mistral-7B | 1/16 | top32_share | 263 | 0.668 | 0.580 |
| Mistral-7B | 1/16 | snapkv@0.0625 | 263 | 0.147 | 0.532 |
| Mistral-7B | 1/16 | h2o@0.0625 | 263 | 0.086 | 0.493 |
| Mistral-7B | 1/16 | obcache_v@0.0625 | 263 | 0.254 | 0.609 |
| Mistral-7B | 1/16 | obcache_k@0.0625 | 263 | 0.169 | 0.590 |
| Mistral-7B | 1/16 | obcache_vk@0.0625 | 263 | 0.235 | 0.586 |
| Mistral-7B | 1/16 | act_change@0.0625 | 263 | 0.587 | 0.509 |
| Mistral-7B | 1/16 | kl@0.0625 | 263 | 0.215 | 0.505 |
| Mistral-7B | 1/4 | D (logged) | 124 | 0.959 | 0.634 |
| Mistral-7B | 1/4 | a_early | 124 | 0.821 | 0.472 |
| Mistral-7B | 1/4 | top32_share | 124 | 0.455 | 0.385 |
| Mistral-7B | 1/4 | snapkv@0.25 | 124 | 0.521 | 0.499 |
| Mistral-7B | 1/4 | h2o@0.25 | 124 | 0.441 | 0.401 |
| Mistral-7B | 1/4 | obcache_v@0.25 | 124 | 0.640 | 0.586 |
| Mistral-7B | 1/4 | obcache_k@0.25 | 124 | 0.635 | 0.540 |
| Mistral-7B | 1/4 | obcache_vk@0.25 | 124 | 0.636 | 0.527 |
| Mistral-7B | 1/4 | act_change@0.25 | 124 | 0.741 | 0.399 |
| Mistral-7B | 1/4 | kl@0.25 | 124 | 0.057 | 0.571 |
| Qwen2.5-14B | 1/16 | D (logged) | 141 | 0.879 | 0.474 |
| Qwen2.5-14B | 1/16 | a_early | 141 | 0.848 | 0.285 |
| Qwen2.5-14B | 1/16 | top32_share | 141 | 0.711 | 0.415 |
| Qwen2.5-14B | 1/16 | snapkv@0.0625 | 141 | 0.172 | 0.613 |
| Qwen2.5-14B | 1/16 | h2o@0.0625 | 141 | 0.524 | 0.551 |
| Qwen2.5-14B | 1/16 | obcache_v@0.0625 | 141 | 0.145 | 0.494 |
| Qwen2.5-14B | 1/16 | obcache_k@0.0625 | 141 | 0.162 | 0.565 |
| Qwen2.5-14B | 1/16 | obcache_vk@0.0625 | 141 | 0.161 | 0.565 |
| Qwen2.5-14B | 1/16 | act_change@0.0625 | 141 | 0.281 | 0.444 |
| Qwen2.5-14B | 1/16 | kl@0.0625 | 141 | 0.657 | 0.686 |
| Qwen2.5-14B | 1/4 | D (logged) | 65 | 0.959 | 0.518 |
| Qwen2.5-14B | 1/4 | a_early | 65 | 0.844 | 0.284 |
| Qwen2.5-14B | 1/4 | top32_share | 65 | 0.503 | 0.316 |
| Qwen2.5-14B | 1/4 | snapkv@0.25 | 65 | 0.122 | 0.554 |
| Qwen2.5-14B | 1/4 | h2o@0.25 | 65 | 0.490 | 0.465 |
| Qwen2.5-14B | 1/4 | obcache_v@0.25 | 65 | 0.137 | 0.324 |
| Qwen2.5-14B | 1/4 | obcache_k@0.25 | 65 | 0.602 | 0.550 |
| Qwen2.5-14B | 1/4 | obcache_vk@0.25 | 65 | 0.588 | 0.543 |
| Qwen2.5-14B | 1/4 | act_change@0.25 | 65 | 0.781 | 0.531 |
| Qwen2.5-14B | 1/4 | kl@0.25 | 65 | 0.794 | 0.571 |

A pooled AUC well below 0.5 means the opposite orientation predicts pooled harm. That is a secondary reading, outside the pre-registered rule.

### Secondary: b = 1/16 versions of the signals as gates, both orientations, D's closed count

| model | signal | gain, low closes | MK3 closed | gain, high closes | MK3 closed | D gain |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | snapkv@0.0625 | +0.011 | 0 | +0.030 | 0 | +0.121 |
| Qwen2.5-1.5B | h2o@0.0625 | +0.011 | 0 | +0.030 | 0 | +0.121 |
| Qwen2.5-1.5B | obcache_v@0.0625 | +0.018 | 4 | +0.008 | 0 | +0.121 |
| Qwen2.5-1.5B | obcache_k@0.0625 | +0.013 | 1 | +0.027 | 0 | +0.121 |
| Qwen2.5-1.5B | obcache_vk@0.0625 | +0.015 | 3 | +0.019 | 0 | +0.121 |
| Qwen2.5-1.5B | act_change@0.0625 | +0.011 | 5 | +0.075 | 57 | +0.121 |
| Qwen2.5-1.5B | kl@0.0625 | +0.053 | 38 | +0.026 | 12 | +0.121 |
| Qwen2.5-3B | snapkv@0.0625 | +0.195 | 100 | +0.115 | 4 | +0.259 |
| Qwen2.5-3B | h2o@0.0625 | +0.092 | 4 | +0.223 | 100 | +0.259 |
| Qwen2.5-3B | obcache_v@0.0625 | +0.083 | 21 | +0.227 | 81 | +0.259 |
| Qwen2.5-3B | obcache_k@0.0625 | +0.190 | 96 | +0.119 | 6 | +0.259 |
| Qwen2.5-3B | obcache_vk@0.0625 | +0.177 | 88 | +0.133 | 15 | +0.259 |
| Qwen2.5-3B | act_change@0.0625 | +0.074 | 16 | +0.236 | 88 | +0.259 |
| Qwen2.5-3B | kl@0.0625 | +0.198 | 84 | +0.108 | 16 | +0.259 |
| Mistral-7B | snapkv@0.0625 | +0.028 | 13 | +0.038 | 1 | +0.187 |
| Mistral-7B | h2o@0.0625 | +0.019 | 0 | +0.059 | 13 | +0.187 |
| Mistral-7B | obcache_v@0.0625 | +0.028 | 13 | +0.058 | 0 | +0.187 |
| Mistral-7B | obcache_k@0.0625 | +0.027 | 13 | +0.043 | 0 | +0.187 |
| Mistral-7B | obcache_vk@0.0625 | +0.028 | 13 | +0.056 | 0 | +0.187 |
| Mistral-7B | act_change@0.0625 | +0.045 | 1 | +0.152 | 75 | +0.187 |
| Mistral-7B | kl@0.0625 | +0.018 | 0 | +0.193 | 97 | +0.187 |
| Qwen2.5-14B | snapkv@0.0625 | -0.001 | 0 | +0.065 | 10 | +0.143 |
| Qwen2.5-14B | h2o@0.0625 | +0.071 | 19 | +0.026 | 0 | +0.143 |
| Qwen2.5-14B | obcache_v@0.0625 | -0.004 | 0 | +0.053 | 0 | +0.143 |
| Qwen2.5-14B | obcache_k@0.0625 | +0.007 | 4 | +0.053 | 0 | +0.143 |
| Qwen2.5-14B | obcache_vk@0.0625 | +0.006 | 3 | +0.053 | 0 | +0.143 |
| Qwen2.5-14B | act_change@0.0625 | +0.042 | 0 | +0.063 | 23 | +0.143 |
| Qwen2.5-14B | kl@0.0625 | +0.119 | 39 | +0.014 | 1 | +0.143 |

Not part of the pre-registered rule. Either orientation is chosen with hindsight here.
