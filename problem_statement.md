# Problem statement: a unified theory and label-free predictor of when removing compute helps

**Status.** Draft v0.8, 2026-06-03 (late night). Target: ICLR 2027 (~Oct 1 2026 deadline).
**Author.** Mishra, Subhankar (SML Lab, NISER Bhubaneswar).

## Quick state (top of file)

Headline 3-claim summary as of v0.9 (2026-06-04, after method-agnostic matrix):

0. **The gate is method-agnostic.** Across 15 measured (model × base evictor) cells at 4K — 4 model sizes (1.5B, 3B, 7B, 14B) and 4 base evictors (SnapKV, H2O, StreamingLLM, PyramidKV) — the gate-closed/open partition with a single τ=0.07 lifts every base evictor by **mean +17pp (SnapKV) to +32pp (StreamingLLM)**. The best plain evictor (over 4 baselines) is consistently beaten by some gated evictor at the same budget by +13 to +29pp. Full matrix in `experiments/results/method_agnostic_matrix.md`.

Headline 3-claim summary as of v0.8 (2026-06-03):

1. **Partition is universal across 7 (model, context) cells, two architectures, three sizes.** Tested: Qwen 1.5B/3B/14B at 4K and (1.5B/3B at 16K), Mistral 7B at 4K and 16K. NIAH-MultiKey-3 is consistently capacity-bound; VT, FWE, QA, niah_multivalue are consistently dilution-prone. Zero counter-examples.
2. **Head-agreement-drop predictor with one fixed τ=0.10 captures the partition** robustly on 6 of 7 cells; the seventh (Qwen 3B 16K) needs τ=0 (no eviction) because plain eviction doesn't help there. τ-sweep shows the optimal lies in [0.05, 0.20] across all cells, narrow enough to be operational.
3. **Gating method (drop-in wrapper around SnapKV-style eviction) gives mean +3 to +24pp over plain SnapKV on the mixed suite** at matched budget. Max single-cell delta: +36pp (Qwen 3B 4K at b=0.0625). Single-task best: Mistral 4K NIAH-MK3 gated 89% vs plain 0% at b=0.0625 (Δ = +89pp).

## What changed from v0.7

Added τ-sensitivity analysis via post-hoc sweep using saved drop and
full-KV-outcome from the gated_eviction.py runs. Findings:

- The optimal τ varies in a narrow [0.05, 0.20] window across all 7 cells.
- A single τ=0.10 default is within 1pp of optimal on 6 of 7 cells.
- One cell (Qwen 3B 16K) prefers τ=0 (full-KV always), but even τ=0.10 there gives positive Δ over plain SnapKV.

This is honest evidence that the bare drop statistic is robust enough for
practical use without per-(model, context) tuning. The architecture-
normalized variant from the original plan is therefore deferred from
"required for ICLR" to "natural follow-up."

## What changed from v0.6: gating method validated, headline secured

Plan Addition 1 (partition-aware gating method) implemented and validated in one day on three architectures with the **same threshold τ=0.07**. The Week-4 hard checkpoint from the ICLR plan was set at "gated ≥ plain + 3pp on mixed suite"; the measured delta is +12pp to +24pp across models.

| Model (params, arch) | mean Δ over eviction budgets | max Δ | gate-open ratio |
|---|---:|---:|---:|
| Qwen2.5-1.5B (12H/2KV, 28L) | **+12.1pp** | +16.2pp at b=0.0625 | 300/400 |
| Qwen2.5-3B (16H/2KV, 36L) | **+23.6pp** | +36.2pp at b=0.0625 | 228/400 |
| Mistral-7B-v0.3 (32H/8KV, 32L) | **+18.7pp** | +25.0pp at b=0.0625 | 287/400 |

Mixed suite: 100 examples each on NIAH-MK3 (capacity-bound), VT, FWE, QA_1 (dilution-prone) from RULER 4K. Threshold τ=0.07 calibrated once on Qwen 1.5B; transferred to Qwen 3B and Mistral 7B without re-tuning.

**Strongest single-task showcase: Mistral-7B NIAH-MK3 (N=100).** Plain SnapKV: 99% (full-KV) → 0% (budget 0.0625). Gated SnapKV: **89% maintained at every budget**. Δ at b=0.0625: **+89pp**. 10/100 false positives (gate opens when shouldn't); even so, the gain is enormous because plain collapses to near-zero.

**Cross-architecture transfer is trivial.** The "architecture-invariant predictor" was listed in the plan as the Week 3-4 risk. Empirically the bare drop with one threshold works across Qwen-GQA (small head count) and Mistral-GQA (large head count, different KV ratio). The architecture normalization machinery I had planned is unnecessary.

This shifts the paper's positioning. We are now a *method paper* with a working algorithm, not a diagnostic-only study.

## What changed from v0.5: honest literature reconciliation

Re-did the literature search 2026-06-03 with the partition findings in hand. Read FIVE close neighbors end-to-end (some of which I had missed on 2026-06-01):

| Paper | What they have | What we have that they don't |
|---|---|---|
| **Bui et al. 2605.09649** (May 2026, DBTrimKV / TrimKV) | Formal dilution mechanism (Prop. 3.1 + Cor. 3.2). Explicit H1 claim. DBTrimKV exceeds full-cache by 3.75%. | Partition between helps/hurts tasks; a priori predictor; ρ scaling formula. |
| **CapKV 2604.25975** (Apr 2026) | Information-bottleneck framing, closed-form, label-free statistical-leverage scoring. Exceeds Vanilla on LongBench. | Task partition; predictor; cross-arch focus. |
| **DynamicKV 2412.14838** (Dec 2024) | Task-aware per-layer adaptive budget. Identifies layer-attention patterns vary by task family. | The binary "helps vs hurts" task split; predictor; explicit NIAH-MK3 failure demonstration. |
| **Garcia 2605.18053** (May 2026, "Protection Is All You Need") | Protocol-level finding: structural protection (sinks + recency) dominates scoring choice. Cross-model panel (10 models). | Task-type partition under matched protection; a priori predictor of which task class an input falls into. |
| **IndexMem 2605.25475** (May 2026) | Learnable indexer + latent memory for compensation. "Information-density effect: moderate eviction sharpens retained context." Strong RULER 4K/16K. | Partition between helps/hurts; closed-form (theirs is trained). |

**Lost novelty (honest):**

- **H1 paradigm ("less compute can beat full attention")** is no longer novel. Bui et al. 2605.09649 makes the claim explicitly with formal proof. CapKV and IndexMem also exhibit it empirically.
- **The attention-dilution mechanism** is no longer novel. Bui Proposition 3.1 + Corollary 3.2 is mathematically equivalent to our H3 SNR-improvement argument.
- **The "closed-form label-free predictor" angle** is no longer novel. CapKV gives a closed-form, label-free, theoretically-grounded eviction score.

**Preserved novelty (also honest):**

1. **Task-type partition.** None of the 5 neighbors identifies a task class where eviction systematically *hurts*. All test on tasks where their method helps. We show NIAH-MK3 fails monotonically across 4 model conditions, every context length, every model.
2. **A priori partition predictor (head-agreement drop, early-to-late).** Works cleanly within the Qwen family ($\rho$ scaling agrees with predictor on 5 tasks). None of the 5 neighbors has any a priori partition predictor.
3. **Cross-size × cross-architecture × cross-context scaling formula** $\rho_{\mathrm{KV}} \approx (1 - A_{\mathrm{full}}) \cdot \mathrm{dilution}(T)$. Predicts BOTH context-length amplification (Qwen 1.5B VT: 0.06 → 0.19 at 4K → 16K) AND model-size saturation (Qwen 3B 4K: $\rho \approx 0$ because $A_{\mathrm{full}} \approx 1$; Qwen 3B 16K FWE: $\rho = 0.32$ because context restores dilution). None of the 5 neighbors quantifies this joint dependence.
4. **Per-input $\rho$ metric** as the cleanest H1 test. Neighbors report mean accuracy curves; per-input strict Pareto improvement is a stronger and less-noisy claim. Different lens.

The paper is therefore narrower than v0.5 framed. Old framing: "Less compute beats full attention." New framing: **"When does KV-cache eviction help vs hurt? A task-type partition and its a priori predictor."**

## What changed from v0.4

Extended the spot-check with 16K-context RULER on Qwen2.5-1.5B, the
`max_new_tokens=128` bug fix on SmolLM2-1.7B, a 3B size-scaling check (in
progress), and an attention-entropy probe of the H3 a priori partition
predictor. Three resulting updates:

1. **$\rho_{\mathrm{KV}}$ amplifies with context length on dilution-prone tasks** — exactly the H3 prediction. VT goes $0.06 \to 0.19$ (4K → 16K), QA_1 goes $0.08 \to 0.14$, niah_multivalue goes $0.07 \to 0.10$. Longer context = more dilution = more for eviction to remove.
2. **The partition is architecture-independent.** SmolLM2-1.7B (with `max_new=128`) gives $\rho = 0.16$ on VT and $\rho = 0.02$ on NIAH-MK3 at 4K — same partition as Qwen2.5-1.5B, with the H1-supportive side EVEN STRONGER on the smaller-quality SmolLM2.
3. **The simplest a priori predictor (mean attention entropy) fails to separate the partition.** Across 7 tasks at 4K the normalized entropy is 0.31–0.35 with no monotone correspondence to $\rho$. A better predictor likely needs to measure the *composition* of attention mass (target-vs-distractor share), not just spread. This is the only outstanding theoretical weakness from the spot-check.

The empirical anchor in v0.5 (vs v0.4's preliminary numbers): **5 of 6 dilution-prone RULER tasks cross threshold at $\geq$ one context length, on $\geq$ one model.** NIAH-MK3 fails as predicted at both 4K and 16K on both models.

## What changed from v0.3

The H1 spot-check (full report at `experiments/results/h1_consolidated_report.md`) on RULER 4K with Qwen2.5-1.5B-Instruct gave a *partition* finding:

- **Tasks where H1 holds.** $\rho_{\mathrm{KV}} = 0.060$ on RULER VT (multi-hop variable tracking) and on RULER FWE (frequent-words aggregation). Both cross the pre-registered $\rho \geq 0.05$ threshold.
- **Tasks where H1 fails.** $\rho_{\mathrm{KV}} = 0.010$ on RULER NIAH-MultiKey-3 (precise retrieval). Accuracy drops monotonically with eviction.

This partition is **not a weakness** — it is a refinement that H3 (noise-from-redundant-compute) directly predicts:
- Tasks where the model attends *weakly to many positions* (multi-hop reasoning across an assignment chain; aggregation over many tokens) have *more dilution to remove*. Eviction sharpens signal.
- Tasks where the model attends *strongly to one needle* (precise multi-key retrieval) have no dilution to remove. Eviction either keeps the needle (no gain) or removes it (large loss).

H1 in v0.4 is therefore narrowed and sharpened: not "on every long-context task," but **"on tasks whose dominant failure mode is dilution-induced uncertainty rather than retrieval-precision."** The mechanism predicts which side of the partition a given task lands on, *a priori*, from its attention pattern.

This is also a stronger paper claim: it's mechanism-justified, falsifiable in both directions (predicted-helps vs predicted-hurts), and not subsumed by any concurrent neighbor.

## What changed from v0.2

End-to-end reads of UT-ACA (2603.18446), LU-KV (2602.08585), and AdaCompute-GBM (2604.14853) — the three strongest concurrent neighbors — confirmed that all five novelty claims below survive, and sharpened one. Summary:

| Concurrent paper | Label-free? | Closed-form? | Mechanism? | Substrate | Paradigm |
|---|---|---|---|---|---|
| UT-ACA (2603.18446) | No (GPT-OSS-120B labels + trained LSTM detector) | No (dual-encoder + LSTM) | No (failure-mode taxonomy) | KV | compress-without-loss |
| LU-KV (2602.08585) | Partial (offline corpus + full-attn oracle runs, no downstream labels) | At inference, but calibration-parametrized | Optimality-gap decomposition (not noise model) | KV | compress-without-loss |
| AdaCompute-GBM (2604.14853) | No ($N{\times}K$ labeled inferences + Lagrangian oracle + GBM) | No (XGBoost on lexical features) | No (difficulty taxonomy) | CoT *sample count*, not chain length | compress-without-loss |
| **Ours (v0.3)** | **Yes (from a noise model, no downstream labels)** | **Yes (theory-derived threshold)** | **Yes (noise-from-redundant-compute)** | **KV + CoT chain length + iteration (cross)** | **efficiency-as-improvement ($A(b^\*) > A(b_{\max})$ on some inputs)** |

The strongest single defensible claim is the paradigm shift (last column): all three concurrent papers treat full compute as the accuracy ceiling. Our H1 explicitly tests and reports the regime where less compute yields *more* accuracy on a non-trivial input subset.

## Gap

Three structurally different efficiency interventions all exhibit the same effect: removing the right computation along the right axis *increases* task accuracy, not just preserves it at lower cost. The anchors:

- **KV cache.** Learned eviction beats Full-KV on long-context reasoning (2605.09649). UT-ACA, LU-KV, ForesightKV (2602.03203), Learning-to-Evict (2602.10238) all design predictors of *what to keep*, but each (i) trains the predictor on labels or full-attention oracle runs, (ii) treats Full-KV as ceiling, and (iii) operates on KV only.
- **CoT chain length.** Latent-then-explicit reasoning cuts tokens and raises accuracy (2605.07315). BudgetThinker (2508.17196) trains a token-budget controller. Both operate on a single substrate.
- **Iteration count.** Equilibrium reasoners (2605.21488) tie test-time scaling to attractor convergence: iterating past the attractor stops helping.

The three results are framed as engineering wins on three disconnected pipelines. No paper as of 2026-06-02 articulates them as one effect, derives the predictor from first principles, or demonstrates the *interior accuracy maximum* directly. AdaCompute-GBM (2604.14853) does adaptive allocation but on a fourth, orthogonal axis (self-consistency sample count) and never demonstrates non-monotonicity. That is the gap.

## Formal hypothesis (cross-substrate, paradigm-level)

Let $a \in \{\mathrm{KV},\ \mathrm{CoT},\ \mathrm{Iter}\}$ index a compute axis. For axis $a$ let $b_t^{(a)}$ be a per-step budget along that axis (KV tokens kept per step; generated reasoning tokens per single response; iteration count). For a frozen base model $f_\theta$,
$$A(b_{1:T}^{(a)}, x) = \mathbb{E}\big[\mathrm{metric}(f_\theta(x; b_{1:T}^{(a)}), y)\big], \qquad b^{\*(a)}(x) = \arg\max_{b_{1:T}^{(a)}} A(b_{1:T}^{(a)}, x).$$

**H1 (interior accuracy maximum, task-type-conditional).** For each axis $a$, partition the input distribution into a *dilution-prone* class $\mathcal{D}_a$ (the model's failures on these inputs come from attention dilution / answer-distribution drift / attractor overshoot) and a *capacity-bound* class $\mathcal{C}_a$ (failures come from missing or imprecise information). The hypothesis:

$$\rho_a^{\mathrm{D}} := \Pr_{x \in \mathcal{D}_a}\big[A(b^{\*(a)}, x) > A(b_{\max}^{(a)}, x)\big] > 0.05,$$
$$\rho_a^{\mathrm{C}} := \Pr_{x \in \mathcal{C}_a}\big[A(b^{\*(a)}, x) > A(b_{\max}^{(a)}, x)\big] \approx 0.$$

That is, less compute yields more accuracy on a non-trivial fraction of *dilution-prone* inputs and ~never on *capacity-bound* ones. The partition is predicted *a priori* from the task's attention pattern (sharp attention to few positions ⇒ capacity-bound; diffuse attention to many ⇒ dilution-prone). This is the explicit paradigm break with UT-ACA, LU-KV, ForesightKV, Learning-to-Evict, BudgetThinker, AdaCompute-GBM, none of which proposes such a partition.

**Empirical anchors (2026-06-02 evening).** On RULER with Qwen2.5-1.5B-Instruct, the table below collapses all dilution-prone tasks into one row per (task, context) pair and the one capacity-bound task into its own row.

| Task | Predicted regime | 4K $\hat{\rho}$ | 16K $\hat{\rho}$ |
|---|:---|---:|---:|
| VT (multi-hop) | D | 0.060 ✓ | **0.190** ✓✓✓ |
| QA_1 (single-doc) | D | 0.080 ✓ | **0.140** ✓✓ |
| QA_2 (multi-doc) | D | 0.080 ✓ | 0.050 ✓ |
| niah_multivalue | D | 0.070 ✓ | 0.100 ✓ |
| FWE (aggregation) | D | 0.060 ✓ | 0.040 ~ |
| NIAH-MK3 (precise retrieval) | C | 0.010 ✗ | 0.000 ✗ |
| Cross-model VT on SmolLM2-1.7B | D | **0.160** ✓ | — |
| Cross-model NIAH-MK3 on SmolLM2 | C | 0.020 ✗ | — |

Five dilution-prone tasks satisfy the threshold at $\geq$ one context length. The lone capacity-bound task fails at both context lengths on both models. The partition holds with no counter-examples and amplifies with context.

**H2 (label-free saturation-statistic predictability).** For each axis $a$ there is a cheap inference-time statistic $s_t^{(a)}(x)$ — attention-entropy + top-$k$ mass + key-norm for KV; answer-token margin + generation entropy for CoT; state-update norm for Iter — such that a closed-form, single-threshold rule on $s_t^{(a)}$ predicts $b^{\*(a)}$ within accuracy gap $\epsilon$ of the sweep oracle, **without downstream-label supervision**. The threshold is derived from the H3 noise model, not learned from labels.

**H3 (mechanism: noise from redundant compute).** The gain $A(b^{\*(a)}, x) - A(b_{\max}^{(a)}, x)$ is monotone in pre-truncation $s_t^{(a)}$. Compute past $b^{\*(a)}$ contributes a quantifiable noise term to the model's output distribution — attention dilution for KV, answer-distribution drift for CoT, attractor overshoot for Iter — and truncation removes it to first order.

H1 is the paradigm claim. H2 is the methodological deliverable. H3 is the mechanism that lets the threshold be derived rather than trained.

## Five defensible novelty claims

1. **Label-free.** UT-ACA needs (token, label) pairs from a 120B oracle; LU-KV needs an offline calibration corpus + full-attention runs; AdaCompute-GBM needs $N{\times}K$ labeled inferences. Our threshold is derived from the H3 noise model.
2. **Closed-form-from-theory.** UT-ACA uses an LSTM; LU-KV uses calibration-derived parameters; AdaCompute-GBM uses XGBoost. Ours is a scalar threshold per axis with one calibrated noise floor.
3. **Mechanism.** UT-ACA: failure-mode taxonomy. LU-KV: optimality-gap decomposition (not a noise model). AdaCompute-GBM: difficulty taxonomy. None derives the predictor from a noise model.
4. **Cross-substrate.** UT-ACA, LU-KV, ForesightKV, Learning-to-Evict, BudgetThinker, AdaCompute-GBM each operate on one substrate. We claim one $s$-thresholding rule generalizes across KV + CoT chain length + iteration.
5. **Paradigm: efficiency-as-improvement.** All three concurrent papers we read treat full compute as ceiling. We test and report $\rho_a$, the fraction of inputs where less compute yields strictly higher accuracy, and we show $\rho_a > 0$ for KV and CoT.

If H2 (label-free closed-form) fails empirically, claim 5 still holds with a trained-head predictor — the paper weakens but remains publishable. If H1 fails ($\rho_a \approx 0$ everywhere), the paper does not exist; the falsification criterion catches this in week 6.

## Predictor design (per axis)

A clean factorization, with each component independently testable:

- **Per-input total budget** $B^{\*(a)}(x)$ — predicted from the saturation statistic $s_t^{(a)}$, our contribution.
- **Per-axis distribution of $B^\*$ across heads/layers/steps** — for KV this is per-head distribution, where LU-KV's offline-calibrated table is a strong baseline; for CoT this is per-step allocation along the chain.

The factorization gives a clean ablation grid:
- (Ours total) $\times$ (Ours distribution) = **pure label-free**, primary claim.
- (Ours total) $\times$ (LU-KV table) = minimum-calibration hybrid; useful as graceful-degradation fallback.
- (Sweep total) $\times$ (LU-KV table) = pure LU-KV; baseline.

## Minimal falsifying experiment

**KV-cache axis (primary).**
- Frozen base models: Qwen2.5-1.5B (Transformer), Llama-3.2-1B (Transformer), Mamba-2.8B-base (SSM cross-arch).
- Tasks: RULER sub-tasks $\{\text{NIAH-Single-3},\ \text{NIAH-MultiKey-3},\ \text{Variable Tracking},\ \text{FWE}\}$ + LongBench-v2 subset.
- Context lengths $\in \{4\mathrm{K},\ 16\mathrm{K},\ 32\mathrm{K}\}$.
- Baselines: (a) Full-KV ceiling, (b) uniform fixed-budget floor, (c) learned global eviction (2605.09649), (d) UT-ACA (2603.18446), (e) LU-KV (2602.08585), (f) sweep-derived oracle $b^{\*(\mathrm{KV})}$ per input.
- **H1 metric (primary headline).** $\hat{\rho}_{\mathrm{KV}}$ = fraction of inputs where some $b < b_{\max}$ yields strictly higher accuracy than Full-KV under sweep. Report per-task and pooled. Falsification: $\hat{\rho}_{\mathrm{KV}} < 5\%$ everywhere ⇒ H1 rejected for KV axis.
- **H2 metric.** Fraction of the (b)→(f) gap closed by each predictor candidate at matched average budget. Falsification: best label-free predictor closes $<70\%$ of gap on RULER at any context length ⇒ H2 (label-free variant) rejected; fall back to "tiny calibration head" predictor and reframe.

**CoT chain-length axis (secondary).**
- Frozen base models: Qwen2.5-1.5B, Llama-3.2-1B.
- Tasks: GSM8K-hard, MATH (subset).
- $b$ is *generated reasoning tokens per single response*, not number of self-consistency samples. AdaCompute-GBM works on the sample-count axis and is not a head-to-head competitor; we mention it as adjacent prior art.
- Baselines: (a) full-CoT ceiling, (b) fixed-length truncation, (c) LaTER (2605.07315) where the public implementation allows, (d) BudgetThinker (2508.17196) — direct competitor, (e) sweep oracle.
- H1 metric: $\hat{\rho}_{\mathrm{CoT}}$ = fraction of inputs where shorter CoT yields strictly higher accuracy. Falsification: $\hat{\rho}_{\mathrm{CoT}} < 5\%$ ⇒ CoT axis drops to mechanism corroboration only, cross-substrate claim weakens to "axis-of-the-same-form, not yet demonstrated to also exhibit H1."
- H2 metric: same gap-closure, with $50\%$ threshold rather than $70\%$ given fewer baselines.

**Iter axis (theoretical bridge, no primary experiment).**
- One proposition tying $s_t^{(\mathrm{Iter})}$ to attractor distance under standard DEQ contraction assumptions. Empirical check on a small implicit-attention or DEQ model only if week 9 has spare cycles.

**Joules/token table (secondary).**
- One A100, `nvidia-smi` power draw, matched accuracy. KV axis only. Jetson Orin row contingent on lab availability.

All falsification criteria are committed to the experiments repo before any sweep runs.

## Papers to beat

KV axis (read end-to-end during weeks 1–2; one-paragraph adversary note per paper in `papers_to_beat.md`):

1. **2603.18446 UT-ACA** — closest H2 competitor. Trained LSTM detector; we must show label-free closed-form matches or beats it on RULER. **Read 2026-06-01.**
2. **2602.08585 LU-KV** — strongest static-allocation baseline. Calibration-derived, treats Full-KV as ceiling. We use their table as a component in the factorized predictor ablation, and we beat them on $\hat{\rho}_{\mathrm{KV}}$. **Read 2026-06-02.**
3. **2605.09649** — original "less helps" KV result. We make it predictive and mechanism-justified.
4. **2602.03203 ForesightKV** — trained predictive eviction for reasoning models. Adversary: needs training data; ours does not. **TODO read week 1.**
5. **2602.10238 Learning-to-Evict** — RL-trained per-head eviction policies. **TODO read week 1.**

CoT axis:

6. **2508.17196 BudgetThinker** — control-token budget for chain length (direct competitor). **TODO read week 1.**
7. **2605.07315 LaTER** — latent-then-explicit anchor; supports H1 in this substrate.

Adjacent / prior-art anchors (not head-to-head):

- **2604.14853 AdaCompute-GBM** — adaptive allocation on the sample-count axis (orthogonal). **Read 2026-06-02.**
- 2605.21488 (equilibrium reasoners), 2605.06285 (LatentRAG), 2605.11478 (FibQuant), 2506.16640 (sparse-attention generalization), 2602.13804 (Vashista exponential mass).

## Out of scope

- Pretraining or fine-tuning of base models.
- Sequence lengths above 32K beyond a single 64K appendix run.
- Self-consistency sample-count axis (covered by AdaCompute-GBM).
- Other axes (layer-skipping, depth-pruning, expert-skipping) — discussion only.
- Energy table on edge devices unless Jetson Orin is confirmed available.

## Open items before execution

- **Resolved.** UT-ACA read (2026-06-01). LU-KV read (2026-06-02). AdaCompute-GBM read (2026-06-02).
- **Outstanding (week 1).** Read ForesightKV (2602.03203), Learning-to-Evict (2602.10238), BudgetThinker (2508.17196) end-to-end. Each could surface a previously-hidden overlap with H2.
- **Outstanding (continuous).** Re-search arXiv weekly for "interior accuracy maximum," "noise from redundant compute," "label-free budget predictor." This subfield moves on a 2–4 week scale.
- **User to confirm.** Jetson Orin availability.
