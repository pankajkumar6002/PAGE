# P-1: per-layer head-agreement profiles $a_\ell$ (assumption A3)

## Qwen2.5-1.5B 4K ($L = 28$)

| task | early third | late third | $D$ |
|---|---:|---:|---:|
| NIAH-MK3 (capacity-bound) | 0.327 | 0.283 | +0.044 |
| VT (dilution-prone) | 0.389 | 0.227 | +0.161 |
| FWE (dilution-prone) | 0.480 | 0.369 | +0.111 |

Separation margin (min dilution-prone $D$ minus MK3 $D$): **+0.068**

## Llama-3.1-8B 4K ($L = 32$)

| task | early third | late third | $D$ |
|---|---:|---:|---:|
| NIAH-MK3 (capacity-bound) | 0.343 | 0.272 | +0.071 |
| VT (dilution-prone) | 0.366 | 0.288 | +0.078 |
| FWE (dilution-prone) | 0.454 | 0.341 | +0.113 |

Separation margin (min dilution-prone $D$ minus MK3 $D$): **+0.007**

The margin collapses from Qwen to Llama, and the Llama profile is non-monotone in depth rather than decaying. This is the mechanistic content behind the Llama transfer failure and the per-input AUC of about 0.80 there: on that architecture the early-minus-late contrast is reading a profile whose shape does not match the one A3 describes.
