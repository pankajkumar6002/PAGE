# P1-7: layer-subsampling ablation for the gate signal D

## Qwen2.5-1.5B 4K (28 layers, 600 rows)

Validity: full-$L$ D recomputed from the per-layer log vs the `drop` the gate thresholded.

The Qwen cells agree bit-exactly. Mistral agrees to $4\times 10^{-4}$, consistent with rounding in the stored per-layer values and with the documented mid-run truncation of its qa\_1 rows. Both are far below $\tau = 0.07$, so neither can flip a gate decision; we flag only differences that could.

| task | $N$ per-layer | $N$ gated | recomputed D | deployed drop | delta |
|---|---:|---:|---:|---:|---:|
| fwe | 100 | 100 | 0.1113 | 0.1113 | 0.00e+00 |
| niah_multikey_3 | 100 | 100 | 0.0437 | 0.0437 | 0.00e+00 |
| qa_1 | 100 | 100 | 0.2362 | 0.2362 | 0.00e+00 |
| vt | 100 | 100 | 0.1613 | 0.1613 | 0.00e+00 |

| layer subset | #layers | mean D (MK3) | min mean D (others) | margin | order kept |
|---|---:|---:|---:|---:|---|
| all 28 layers | 28 | 0.0437 | 0.1113 | +0.0676 | yes |
| every 2nd | 14 | 0.0206 | 0.0726 | +0.0520 | yes |
| every 4th | 7 | 0.0888 | 0.0872 | -0.0017 | **NO** |
| early bin only | 9 | 0.0677 | 0.0267 | -0.0410 | **NO** |
| first+last layer | 2 | 0.0010 | 0.0670 | +0.0660 | yes |

## Qwen2.5-3B 4K (36 layers, 600 rows)

Validity: full-$L$ D recomputed from the per-layer log vs the `drop` the gate thresholded.

The Qwen cells agree bit-exactly. Mistral agrees to $4\times 10^{-4}$, consistent with rounding in the stored per-layer values and with the documented mid-run truncation of its qa\_1 rows. Both are far below $\tau = 0.07$, so neither can flip a gate decision; we flag only differences that could.

| task | $N$ per-layer | $N$ gated | recomputed D | deployed drop | delta |
|---|---:|---:|---:|---:|---:|
| fwe | 100 | 100 | 0.0377 | 0.0377 | 0.00e+00 |
| niah_multikey_3 | 100 | 100 | -0.0038 | -0.0038 | 0.00e+00 |
| qa_1 | 100 | 100 | 0.1047 | 0.1047 | 0.00e+00 |
| vt | 100 | 100 | 0.0757 | 0.0757 | 0.00e+00 |

| layer subset | #layers | mean D (MK3) | min mean D (others) | margin | order kept |
|---|---:|---:|---:|---:|---|
| all 36 layers | 36 | -0.0038 | 0.0377 | +0.0415 | yes |
| every 2nd | 18 | 0.0023 | 0.0148 | +0.0125 | yes |
| every 4th | 9 | 0.0331 | 0.0367 | +0.0035 | yes |
| early bin only | 12 | 0.0918 | 0.0638 | -0.0280 | **NO** |
| first+last layer | 2 | 0.0096 | -0.0435 | -0.0532 | **NO** |

## Mistral-7B 4K (32 layers, 578 rows)

> Note: skipped 1 corrupted line(s) in `drops_mistral7b_4k_n100.jsonl` and 0 in `gated_4k_mistral7b.jsonl`. These are NUL-byte runs from an interrupted write in the original logging run, not parse-policy choices.

Validity: full-$L$ D recomputed from the per-layer log vs the `drop` the gate thresholded.

The Qwen cells agree bit-exactly. Mistral agrees to $4\times 10^{-4}$, consistent with rounding in the stored per-layer values and with the documented mid-run truncation of its qa\_1 rows. Both are far below $\tau = 0.07$, so neither can flip a gate decision; we flag only differences that could.

| task | $N$ per-layer | $N$ gated | recomputed D | deployed drop | delta |
|---|---:|---:|---:|---:|---:|
| fwe | 100 | 100 | 0.0805 | 0.0804 | 1.11e-04 (approx) |
| niah_multikey_3 | 100 | 100 | 0.0600 | 0.0600 | 3.20e-05 (approx) |
| qa_1 | 78 | 100 | 0.0951 | 0.0955 | 3.99e-04 (approx), N differs |
| vt | 100 | 100 | 0.0953 | 0.0953 | 1.33e-05 (approx) |

| layer subset | #layers | mean D (MK3) | min mean D (others) | margin | order kept |
|---|---:|---:|---:|---:|---|
| all 32 layers | 32 | 0.0600 | 0.0774 | +0.0174 | yes |
| every 2nd | 16 | 0.0611 | 0.0763 | +0.0152 | yes |
| every 4th | 8 | 0.0078 | -0.0116 | -0.0194 | **NO** |
| early bin only | 10 | -0.0276 | -0.0502 | -0.0225 | **NO** |
| first+last layer | 2 | -0.0595 | -0.2137 | -0.1541 | **NO** |

Reading: the ordering that the gate depends on, NIAH-MK3 smallest, is what each subset must preserve. A positive margin means a threshold separating the capacity-bound task from every other task still exists on that subset.

Head-pair subsampling is not evaluable from these logs: the probe averages over head pairs before writing, so per-pair Jaccards would need a modified probe and a fresh prefill pass.
