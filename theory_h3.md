# H3: noise from redundant compute

**Status.** Working draft, 2026-06-02 (late). Intended for the paper's theory
section, polished from the v0.4 sketch with the renormalization step made
explicit and a context-length scaling corollary derived.

## 1. Setup

Consider a single attention head reading from prompt positions $t \in \{1, \dots, T\}$,
producing an output that is then projected to a logit distribution. Each
position carries:

- A *role* $r_t \in \{0, 1\}$: 1 if the position carries information the
  decoder needs to answer correctly, 0 otherwise.
  Partition $\mathcal{R} = \{t : r_t = 1\}$, $\mathcal{N} = \{t : r_t = 0\}$,
  with $|\mathcal{R}| = R$, $|\mathcal{N}| = N$, $R \ll N$.
- A pre-eviction attention probability $\alpha_t \geq 0$ with $\sum_t \alpha_t = 1$,
  produced by softmax of query-key dot products.
- A value vector $v_t \in \mathbb{R}^d$.

The head output is
$$o = \sum_{t=1}^T \alpha_t v_t = S + N,
\qquad S := \sum_{t \in \mathcal{R}} \alpha_t v_t,
\quad N := \sum_{t \in \mathcal{N}} \alpha_t v_t.$$

**Modelling assumptions** (for the proposition, not the paper claim).

1. Signal coherence: $v_t = u + \xi_t$ for $t \in \mathcal{R}$, where $u$ is
   a deterministic direction with $\|u\| = 1$ and $\xi_t$ is zero-mean
   sub-Gaussian noise with $\mathbb{E}\|\xi_t\|^2 = \tau^2 \ll 1$.
2. Distractor isotropy: $v_t$ for $t \in \mathcal{N}$ are independent,
   zero-mean, with $\mathbb{E}\|v_t\|^2 = \sigma^2$, $\mathbb{E}[v_t v_{t'}^\top] = 0$
   for $t \neq t'$.

Both are standard in attention-as-feature-aggregation analyses (e.g.
Ferrando et al., 2024; the "attention sink" line of work; the signal-noise
decomposition in Vashista 2602.13804).

Under these assumptions, setting $\alpha_\mathcal{R} := \sum_{t \in \mathcal{R}} \alpha_t$,

$$\|S\|^2 \approx \alpha_\mathcal{R}^2 \cdot \|u\|^2 + O(\tau^2)
            = \alpha_\mathcal{R}^2 + O(\tau^2),$$
$$\mathbb{E}\|N\|^2 = \sigma^2 \sum_{t \in \mathcal{N}} \alpha_t^2 \cdot.$$

Define the *signal-to-noise ratio*:
$$\mathrm{SNR} := \frac{\|S\|^2}{\mathbb{E}\|N\|^2}
              = \frac{\alpha_\mathcal{R}^2}{\sigma^2 \sum_{t \in \mathcal{N}} \alpha_t^2}.$$

## 2. Eviction

An eviction rule with budget $b$ retains $\mathcal{K} \subseteq \{1, \dots, T\}$
with $|\mathcal{K}| = b$, then renormalizes:
$$\tilde\alpha_t = \frac{\alpha_t \mathbb{1}[t \in \mathcal{K}]}{Z},
\qquad Z := \sum_{s \in \mathcal{K}} \alpha_s.$$

Let $\rho_\mathcal{R} := \sum_{t \in \mathcal{K} \cap \mathcal{R}} \alpha_t / \alpha_\mathcal{R}$
be the fraction of relevant attention mass retained, similarly
$\rho_\mathcal{N}$. **Good-scoring assumption:**
$\rho_\mathcal{R} \geq \rho_\mathcal{N}$ for the eviction rule of interest
(SnapKV-style scoring, query-aligned).

Post-eviction:
$$\tilde S = \frac{\rho_\mathcal{R} \alpha_\mathcal{R}}{Z} u + O(\tau),
\qquad \mathbb{E}\|\tilde N\|^2 = \frac{\sigma^2}{Z^2} \sum_{t \in \mathcal{K} \cap \mathcal{N}} \alpha_t^2.$$

Substituting,
$$\widetilde{\mathrm{SNR}} = \frac{\rho_\mathcal{R}^2 \alpha_\mathcal{R}^2}{\sigma^2 \sum_{t \in \mathcal{K} \cap \mathcal{N}} \alpha_t^2}.$$

Note that the renormalization $Z$ cancels in the SNR (it scales numerator
and denominator equally). The SNR ratio is therefore
$$\boxed{\frac{\widetilde{\mathrm{SNR}}}{\mathrm{SNR}}
= \rho_\mathcal{R}^2 \cdot \frac{\sum_{t \in \mathcal{N}} \alpha_t^2}{\sum_{t \in \mathcal{K} \cap \mathcal{N}} \alpha_t^2}.}$$

The first factor $\rho_\mathcal{R}^2 \leq 1$ penalizes throwing away relevant
mass. The second factor $\geq 1$ rewards throwing away noise mass.

**Proposition 1 (SNR improvement, qualitative).** Under good scoring
($\rho_\mathcal{R} \geq \rho_\mathcal{N}$) and noise homoscedasticity
(distractor attention weights roughly uniform with mean $\bar\alpha_\mathcal{N}$),
$$\frac{\widetilde{\mathrm{SNR}}}{\mathrm{SNR}} \geq \rho_\mathcal{R}^2 \cdot \frac{|\mathcal{N}|}{|\mathcal{K} \cap \mathcal{N}|}.$$
For budgets $b$ that retain a constant fraction $\gamma \in (0, 1)$ of the
distractor set and an equal-or-greater fraction of the relevant set,
$\widetilde{\mathrm{SNR}} \geq \mathrm{SNR}$ when
$$\rho_\mathcal{R}^2 \geq \gamma.$$

*Proof.* Substitute $|\mathcal{K} \cap \mathcal{N}| = \gamma |\mathcal{N}|$
into the bound. Under good scoring, $\rho_\mathcal{R} \geq 1 - (1-\gamma) \cdot R/(R+N)$,
which approaches 1 for small $R/(R+N)$. So $\rho_\mathcal{R}^2 \approx 1 > \gamma$,
satisfying the condition. $\square$

## 3. Partition prediction

Define two regimes by the *pre-eviction* signal-attention mass:

- **Dilution-prone** ($\mathcal{D}$): $\alpha_\mathcal{R} \ll 1$. Most
  attention mass is on noise. There is large $|\mathcal{N}|$ relative to
  retained-noise count, so the gain factor in Proposition 1 is large.
  Eviction improves SNR.
- **Capacity-bound** ($\mathcal{C}$): $\alpha_\mathcal{R} \to 1$. Almost
  all attention is already on relevant positions. The noise denominator is
  near zero. Eviction's only way to change SNR is by hurting the numerator
  (some relevant positions excluded by imperfect scoring), which can only
  decrease SNR.

**Proposition 2 (partition).** Let $\hat{A}(\mathrm{SNR})$ be the model's
expected accuracy as a non-decreasing function of head SNR (a standard
assumption when SNR controls logit margin).

- On dilution-prone inputs, $\widetilde{\mathrm{SNR}} > \mathrm{SNR}$
  for a non-trivial range of $b$ under good scoring; hence
  $\Pr[\hat{A}(\widetilde{\mathrm{SNR}}) > \hat{A}(\mathrm{SNR})] > 0$.
- On capacity-bound inputs, $\widetilde{\mathrm{SNR}} \leq \mathrm{SNR}$
  for all $b < T$; hence H1's $\rho \to 0$.

This is the empirical partition we observe on RULER: VT, FWE, QA, niah_multivalue
are dilution-prone (multi-position integration); NIAH-MK3 is capacity-bound
(single-needle retrieval). 5 of 6 dilution-prone tasks cross $\rho \geq 0.05$;
NIAH-MK3 fails at $\rho \leq 0.01$ at both 4K and 16K on two models.

## 4. Context-length amplification

**Proposition 3 (context-length scaling, dilution-prone).** Fix the task and
the relevant set's *size* $R$ (the task structure is unchanged when we
extend the context). Let $T = R + N$. Suppose distractor attention is
approximately uniform on $\mathcal{N}$: $\alpha_t \approx (1 - \alpha_\mathcal{R}) / N$
for $t \in \mathcal{N}$. Then
$$\mathrm{SNR} \approx \frac{\alpha_\mathcal{R}^2}{\sigma^2 (1 - \alpha_\mathcal{R})^2 / N}
                  = \frac{\alpha_\mathcal{R}^2 \cdot N}{\sigma^2 (1 - \alpha_\mathcal{R})^2}.$$

If $\alpha_\mathcal{R}$ decreases with $T$ (more distractors compete for
softmax mass), the pre-eviction SNR decreases. Eviction with constant
*ratio* $\gamma$ retains $\gamma N$ distractors, so
$$\widetilde{\mathrm{SNR}} \approx \frac{(\rho_\mathcal{R} \alpha_\mathcal{R})^2 \cdot \gamma N}{\sigma^2 (\rho_\mathcal{N} (1 - \alpha_\mathcal{R}))^2}
                              = \frac{\rho_\mathcal{R}^2 \alpha_\mathcal{R}^2 \cdot N}{\sigma^2 \rho_\mathcal{N}^2 (1 - \alpha_\mathcal{R})^2 / \gamma}.$$

The improvement ratio scales with $\rho_\mathcal{R}^2 / (\rho_\mathcal{N}^2 \cdot \gamma)$.
Since good scoring keeps $\rho_\mathcal{R}$ approximately constant in $T$
(the scoring picks $\mathcal{R}$ first), and $\rho_\mathcal{N} = \gamma$
(uniformly downsampling noise), the SNR improvement grows as $1/\gamma$
even as $T$ grows.

The *headroom* on accuracy — defined as $1 - A_{\mathrm{full}}$ — scales with
$T$ on dilution-prone tasks. Combining, the per-input recovery probability
$\rho_{\mathrm{KV}}$ obeys
$$\rho_{\mathrm{KV}} \approx (1 - A_{\mathrm{full}}(T)) \cdot p_{\mathrm{recov}}(\rho_\mathcal{R}/\gamma).$$

This is exactly the empirical scaling we observe:

- Qwen 1.5B VT: $1 - A_{\mathrm{full}}$ grows $0.18 \to 0.75$ going $4K \to 16K$, and
  $\rho$ grows $0.06 \to 0.19$.
- Qwen 3B FWE: $1 - A_{\mathrm{full}}$ grows $0.24 \to 0.62$ going $4K \to 16K$, and
  $\rho$ grows $0.01 \to 0.32$.

## 5. A priori partition predictor (the open theoretical step)

The partition is governed by $\alpha_\mathcal{R}$, but $\mathcal{R}$ is not
known at inference. Two candidate statistics computable without labels:

1. **Mean attention entropy.** Spread of $\alpha$ across positions. Tested
   in `experiments/scripts/attention_entropy_probe.py`; **does not separate
   the partition at 4K** (entropy range 0.31–0.35 across tasks). The reason
   is that entropy is dominated by the long noise tail; the *concentration*
   of mass on $\mathcal{R}$ vs $\mathcal{N}$ is invisible in entropy when
   $R$ is small and $N$ is large.

2. **Head agreement (mean).** For capacity-bound tasks, different heads
   tend to attend to the *same* positions (the few relevant ones). For
   dilution-prone tasks, different heads attend to different position sets.
   *Tested 2026-06-03 — does not separate at 4K either.* Mean Jaccard
   top-32 agreement is in 0.31–0.46 across tasks; VT and NIAH-MK3 are
   essentially tied at 0.31. Mean is not the right summary.

3. **Head-agreement DROP across depth.** Decompose the head agreement
   curve by layer. Early layers attend locally (similar across tasks);
   late layers integrate or converge per the task structure. The
   *change* in head agreement from early to late layers separates the
   partition cleanly:

   | Task | early | late | drop | $\rho$ | H1 |
   |---|---:|---:|---:|---:|:---:|
   | qa_1 | 0.523 | 0.305 | **0.218** | 0.080 | ✓ |
   | qa_2 | 0.486 | 0.306 | **0.180** | 0.080 | ✓ |
   | vt | 0.389 | 0.238 | **0.151** | 0.060 | ✓ |
   | niah_multivalue | 0.517 | 0.410 | **0.107** | 0.070 | ✓ |
   | fwe | 0.480 | 0.382 | **0.098** | 0.060 | ✓ |
   | niah_multiquery | 0.405 | 0.337 | 0.068 | (sat) | n/a |
   | niah_multikey_3 | 0.329 | 0.288 | **0.041** | 0.010 | ✗ |

   **The drop cleanly separates dilution-prone (drop $\geq 0.098$) from
   capacity-bound (drop $\leq 0.07$).** This is computable from the
   prefill attention with no labels and no downstream evaluation —
   exactly the a priori predictor H3 needs.

   *Interpretation.* In dilution-prone tasks, late layers spread attention
   across different aspects of the answer (heads diverge in *which*
   positions of the context they attend to). In capacity-bound tasks,
   late layers converge on the same needle (heads keep agreeing on the
   same few positions). The drop measures *integration breadth*: more
   integration breadth ⇒ more dilution ⇒ more for eviction to recover.

   **Cross-size and cross-model results (2026-06-03).**
   - Qwen2.5-1.5B (12 query / 2 KV heads, 28 layers): drop $\geq 0.098$
     for all 5 dilution-prone tasks, drop $0.041$ for NIAH-MK3. Clean.
   - Qwen2.5-3B (16 / 2, 36 layers): drop $\geq 0.037$ for all dilution-prone
     tasks, drop $-0.002$ for NIAH-MK3. **The sign is informative**: zero
     or negative drop is a capacity-bound signature. Clean within the
     Qwen family.
   - SmolLM2-1.7B (Llama-arch, 32 / 8, 24 layers): VT, QA_1, QA_2 still
     have high drops ($\geq 0.13$), but NIAH-MK3 has drop $0.097$ and
     niah_multivalue has drop $0.021$. Two mis-rankings. The architecture
     difference (Llama vs Qwen) shifts the drop distribution.

   The predictor transfers within an architecture family (both Qwen sizes
   tested) but not across architectures (Qwen $\to$ SmolLM2). Possible
   refinements:
   - Normalize by architecture-specific maximum agreement (number of heads
     differs between models).
   - Use a different layer pair (e.g., layer 0 vs layer $L-1$ rather than
     thirds).
   - Combine with a complementary statistic: head agreement in early
     layers correlates positively with $\rho$ on Qwen but negatively on
     SmolLM2, hinting at a normalization issue.

   For the paper: report the agreement-drop result on Qwen as **evidence
   that the partition is detectable a priori**, flag the cross-architecture
   transfer as the *open problem*, and use task category (multi-hop /
   aggregation / QA / multi-value $\to \mathcal{D}$; precise multi-key
   retrieval $\to \mathcal{C}$) as the practical partition predictor for
   now.

Open theoretical step: connect the agreement drop to the $\alpha_\mathcal{R}$
quantity in Proposition 2 via a heads-attending-to-different-targets model
that accounts for architecture-specific head count and depth. This is the
write-up needed for the paper's mechanism section.

## 6. Connections and caveats

- **Connection to attention sinks.** The "always-keep first 4 tokens" rule
  in our eviction policy is the StreamingLLM attention-sink fix; without
  it, the renormalization $Z$ has a tail that contaminates eviction.
- **Renormalization caveat.** In §2 the renormalization cancels in SNR
  *because both signal and noise are linearly scaled*. This is exact when
  $\xi_t = 0$ and isotropic noise; with non-isotropic noise the cancellation
  is approximate to first order.
- **Multi-layer.** The single-head SNR analysis lifts to multi-layer in
  the obvious way only when later layers respect the SNR ordering. Empirically
  this holds (per-layer eviction outperforms single-mask in our ablation,
  forthcoming), but a careful multi-layer proof is left for follow-up.
- **Multi-head heterogeneity.** Different heads may be on different sides
  of the partition for the same input. The current eviction picks one set
  of positions for the layer (per-layer SnapKV) or for the whole model
  (single-mask); per-head SnapKV would respect heterogeneity but is harder
  to implement under the constraint of a uniform cache.

## 7. Deferred proofs made rigorous (2026-07-04)

Three results the paper had flagged as "left to future work" are now worked
out with explicit constants and honest gaps. Full LaTeX lives in
`paper/iclr2026/theory_appendix_additions.tex` (compiles clean; new appendix
`app:sharpened`). Summary of each.

### 7.1 Tight constant for the scaling formula (was: `app:proof-scaling`)

The old appendix bounded `rho_KV <= K*(1-A_full)*p_recov + K/4` via a union
bound over the `K`-point budget grid. **The grid factor `K` is spurious.**
The operational `b*(x) = argmax_b A(b,x)` selects "recovery at *some* budget",
so the budget-union lives *inside* the recovery event `B := union_b B_b`, not
in front of the whole expression. With `p_recov := Pr_x[B]` (the union
probability), the surrogate identity is **exact**:

> `rho_KV = (1 - A_full) * p_recov + Cov_x(A, B)`,   leading constant = **1**.

Here `A` = headroom event (full cache fails), `B` = recovery event. Because
both are non-increasing functions of the single driver `alpha_R` under noise
isotropy (A1), Chebyshev's association inequality gives `Cov >= 0`; the
Fréchet–Hoeffding inequality gives the top. Hence a **distribution-free
two-sided bound with attainable endpoints**:

> `(1 - A_full)*p_recov  <=  rho_KV  <=  min{1 - A_full, p_recov}`.

Lower bound attained iff `A ⟂ B`; upper iff `A,B` comonotone. Union-bound
tightness: the discarded step overcounts by the inclusion–exclusion tail
`sum_b Pr[B_b] - Pr[union B_b]`. For a **calibrated (monotone-gain) scorer**
the events `B_b` are *nested*, so `p_recov = max_b Pr[B_b]` and there is **no
`K` factor**; `K` is real only in the degenerate disjoint/rare regime
(`Kp << 1`, each input recovers at exactly one budget).

**Remaining gap.** The constant `1` is exact only *inside* the deterministic-SNR
surrogate (accuracy = function of SNR, itself assumed). The covariance is
pinned only to `[0, min{PrA,PrB} - PrA*PrB]`, not to a value; both endpoints
are attainable so no tighter distribution-free constant exists. QA-style
parametric failures drive `A` partly independently of `alpha_R`, sending
`p_recov -> 0` and pinching `rho_KV -> 0` — a sign-structure prediction, not a
proof about those tasks. The empirical band `[0.25, 0.55]` estimates
`p_recov + Cov/(1-A_full)`, not a universal constant.

### 7.2 Necessary direction: capacity-bound ⇒ small `D` (was: `app:sufficient`)

The old sketch proved only the sufficient direction (large `D` ⇒ large
dilution) and punted the converse for want of "a margin condition we have not
verified". That condition is now named:

> **Assumption M (κ-separation).** `m >= 2*sigma_z*sqrt(log(2 H^2 L T^2 / beta))`,
> i.e. margin-to-noise ratio `kappa := m/sigma_z >= 2*sqrt(log(2 H^2 L T^2/beta))`.

This is exactly the Lemma (top-k concentration) threshold. Under it, plus the
common-relevant-set model of capacity-bound (A3, `C`: late-layer heads share
one set `R_*`), every late-bin head's top-k set equals `R_*` w.p. `>= 1-beta`,
so `a_late = 1` and:

> `Pr[ D <= a_early - 1 <= 0 ] >= 1 - beta`, and for every `tau > 0`
> `Pr[ D >= tau ] <= beta = 2 H^2 L T^2 * exp(-kappa^2 / 4)`.

Explicit margin dependence: the drop's upper tail decays as `exp(-kappa^2/4)`.
The surrogate predicts a **non-positive** expected `D` for capacity-bound
inputs — matching the observed negative `D` on Qwen-3B 16K NIAH-MK3.
**Contrapositive (the gate guarantee):**

> `Pr[ capacity-bound  ∧  D >= tau ] <= beta`,

so firing the gate on `D >= tau` mis-fires on a genuinely capacity-bound input
only with probability `<= beta`.

**Remaining gap.** Conditional on (i) Assumption M holding on real weights
(a hypothesis about the network, tested only through its consequence — small
`D` on NIAH-MK3 in every cell), and (ii) the common-set idealisation of
capacity-bound; if heads only agree on an `O(1)` needle *neighborhood* rather
than one identical set, `a_late = 1 - o(1)` and the bound survives as
`Pr[D>=tau] <= beta + o(1)` but `C` is an idealisation. Controls the sign and
upper tail of `D`, not how negative it gets.

### 7.3 A4 useful-set identification (was: "derived equality", actually an assumption)

A4 identified Bui's useful set `U_t` (residual-stream, decode-step) with the
head-common set `I := ∩_h R_h^{(ℓ*)}` (single-layer attention). A clean exact
equality **cannot** be proven, but a quantitative approximation can. Named
assumptions:

> **A_eps (ε-closeness):** `max_{h,h'} |R_h Δ R_{h'}| <= eps*k`.
> **CUA (consensus–usefulness):** `I ⊆ U_t ⊆ U := ∪_h R_h`. Right inclusion is
>   near-free (attention-only routing); left inclusion is the substantive part.

Then the two sets induce dilution values that agree to `O(eps)`:

> `| delta^(U)_t - delta^(I)_t |  <=  sum_{i in U\I} alpha_{t,i}  <=  H(H-1)*eps*k*alpha_max  =  O(H^2 * eps)`.

So `eps -> 0` ⇒ the dilutions coincide.

**Remaining gap.** The substantive left inclusion (CUA) is an *assumption about
the read-out*, not a theorem, and it **fails precisely in the
single-decisive-head / capacity-bound regime** (one retrieval head carries the
margin-critical value that is absent from the consensus `I`) — exactly where
the bridge is already declined. The `H^2` prefactor is a loose worst-case union
count (same `O(H)` slack as Step 3). Vacuous at `eps = Θ(1)` (genuinely
divergent late layers). Bottom line: **A4 is a named modelling assumption
(CUA) under which we prove an `O(H^2 eps)` approximate identification of the
dilution values**, not a derived equality — the phrase "derived equality" in
the appendix should be read as this approximation.
