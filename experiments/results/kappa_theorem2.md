# T-6: is Theorem 2's margin assumption satisfied?

Theorem 2 bounds `Pr[D >= tau]` for capacity-bound inputs by
`beta = 2 H^2 L T^2 exp(-kappa^2/4)`, informative only when `beta < 1`.

| quantity | value |
|---|---:|
| model / task | Qwen2.5-1.5B, NIAH-MK3, 20 inputs |
| (H, L, T) | (12, 28, ~3956) |
| kappa required for beta < 1 | **10.11** |
| kappa measured (median over ~10744 head-query pairs/input) | **3.33** |
| kappa measured (max over any head) | 77.66 |

**The bound is vacuous at the paper's own (H, L, T).** The measured
margin-to-noise ratio is roughly 3x smaller than the theorem needs,
so Corollary 2, described as "the a-priori guarantee the gate needs", carries no
force at realistic parameters. This is a self-reported negative: a measured
refutation of one's own sufficient condition is a better position than an
unmeasured assumption.

Operationalisation: per (layer, head, observation query), the top-k logits are
treated as the relevant set R and the rest as noise N, with
`m = mean(R) - mean(N)` and `sigma_z = std(N)`. The surrogate does not fix this
split, so the comparison is about order of magnitude, not the third decimal.

Correction applied during measurement: causal masking fills with `finfo.min`
(~-3.4e38), which is finite, so an `isfinite()` filter alone leaves masked keys
in the noise set and inflates both m and sigma_z beyond any physical value.
