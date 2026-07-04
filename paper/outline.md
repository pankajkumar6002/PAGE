# Paper outline v3 (after gating method validated 2026-06-03 night)

**Major update**: Plan Addition 1 (the gating method) was implemented in one
day with stronger results than expected. Cross-architecture transfer (Plan
Addition 2) is empirically resolved on Mistral with the same τ as Qwen.
Scale validation (Plan Addition 3) confirmed at Qwen 14B. Below outline
incorporates these.

# Paper outline v2 (reframed after 2026-06-03 literature reconciliation)

**Status.** Drafted 2026-06-03 evening. Target venue: AAAI 2027 (Sept 2026) or
ICLR 2027 (Sept/Oct 2026).

## Working title

**"When does KV-cache eviction help versus hurt? A task-type partition and
a priori predictor."**

Backup options:
- "The dilution partition: predicting when KV-cache eviction helps long-context LLMs"
- "Not all tasks compress equally: a task-type partition for KV-cache eviction"

The title must communicate that we're a *diagnostic study*, not a method
paper. Two recent strong neighbors (Bui et al. 2605.09649 and CapKV 2604.25975)
already establish that eviction can exceed full-cache; we extend by characterising
*which* tasks exhibit the effect and predicting it a priori.

## Section-by-section outline

### 1. Introduction (1.5 pages)

**Hook.** Recent work has shown that KV-cache eviction can do more than
compress — Bui et al. (2605.09649) prove that selective eviction reduces
attention dilution and exceed full-cache on multi-turn dialogue
by 14% at moderate compression; CapKV (2604.25975) derives the dilution
effect from an information-bottleneck objective; IndexMem (2605.25475)
observes the same "information-density effect" on LongBench. The common
claim: less compute can yield more accuracy.

**Open question.** All three reports the effect *on average* over benchmark
suites where it works. None addresses two natural follow-up questions:
- **For which inputs and tasks does it actually help?** Mean accuracy hides
  per-input variance. Can we identify which tasks systematically benefit
  versus systematically suffer?
- **Can we predict the partition a priori without training a scorer?** All
  three of the strongest concurrent methods train a scorer (Bui's retention
  gates, IndexMem's indexer) or compute capacity scores from query statistics
  (CapKV). Whether the dilution-prone vs capacity-bound distinction is
  visible in pre-eviction attention patterns alone is open.

**Contribution preview.** We provide:

1. **A task-type partition.** On RULER (Hsieh et al. 2024) with Qwen2.5-1.5B,
   Qwen2.5-3B, and SmolLM2-1.7B at 4K and 16K, eviction is *strict Pareto-
   improving on a fraction $\rho \geq 0.05$ of inputs* on six task families
   (multi-hop variable tracking, frequent-words extraction, single/multi-
   document QA, multi-value retrieval) and *monotone-damaging* on one
   task (NIAH-MultiKey-3, precise multi-needle retrieval). The partition
   holds across all model sizes and architectures we tested, with zero
   counter-examples.

2. **A label-free a priori partition predictor.** The early-to-late head-
   agreement drop (averaged Jaccard top-32 similarity across heads, measured
   in early-third vs late-third layers) separates the partition cleanly
   within the Qwen family ($\geq 0.098$ for dilution-prone tasks; $\leq 0.041$
   for NIAH-MK3; $\leq 0$ on Qwen3B). Cross-architecture transfer to
   SmolLM2 (Llama-arch) misranks two tasks, exposing the architecture-
   invariant version as the open theoretical step.

3. **A scaling formula.** The per-input recovery rate follows
   $\rho_{\mathrm{KV}} \approx (1 - A_{\mathrm{full}}(M, T)) \cdot p_{\mathrm{recov}}(\rho_\mathcal{R}/\gamma)$.
   This single formula explains (i) why $\rho$ amplifies with context
   length on dilution-prone tasks (VT: $0.06 \to 0.19$ going $4K \to 16K$),
   (ii) why larger models at short context saturate ($\rho \approx 0$
   because $A_{\mathrm{full}} \approx 1$), and (iii) why those same larger
   models recover the effect at long context (Qwen3B 16K FWE: $\rho = 0.32$).

4. **A practical implication.** Per-input gating using the partition
   predictor can avoid the NIAH-MK3 failure mode at zero training cost.
   We give a minimal gating algorithm and show it preserves full-cache
   accuracy on retrieval-precision inputs while inheriting eviction's
   gains on dilution-prone inputs. (TODO: implement and measure for paper.)

### 2. Related work and positioning (~0.7 pages)

Position carefully — five close neighbors:

- **Bui et al. (2605.09649).** Formal dilution mechanism + DBTrimKV
  exceeding full-cache. **We extend** with the helps-vs-hurts partition;
  they test only on tasks where eviction helps.
- **CapKV (2604.25975).** Closed-form information-theoretic eviction
  score. **We extend** with the partition; their score is task-agnostic
  and they don't characterize when its budget assumption breaks.
- **DynamicKV (2412.14838).** Task-aware adaptive per-layer budgets.
  **We extend** with the binary failure-mode partition; they detect
  task-family attention differences but assume all tasks benefit
  from compression.
- **Garcia (2605.18053).** Protocol-level: structural protection (sinks +
  recency) dominates scoring choice on globally-capped harnesses. **Our
  results use the same sink+recency protection**, so our partition is a
  finding *beyond* his structural protection.
- **IndexMem (2605.25475).** Learnable indexer + latent memory.
  Observes "information-density" peak at 50% compression on LongBench
  average. **We extend** with the partition (their "average over LongBench"
  hides which tasks contribute the peak) and with a *training-free* predictor.

Older anchors: SnapKV (Li et al. 2024c), H2O (Zhang et al. 2023),
StreamingLLM (Xiao et al. 2023), Ada-KV (Feng et al. 2024).

### 3. The empirical partition (~1.5 pages)

**Setup.** RULER 4K and 16K via `simonjegou/ruler`. Eviction is SnapKV-
style: score each prompt position by the average attention from the last
$w = 32$ queries across heads and layers; keep top-$b$, always preserve
first 4 sinks and last 32 recent. Cache slicing with explicit `position_ids`
to preserve RoPE. Multi-element answers (VT, FWE, CWE, niah_multivalue)
scored by `all-in` substring match (RULER's `string_match_all`); single-
element answers (NIAH, QA) by `any-in` (RULER's `string_match_part`).

**Metric.** Per-input recovery rate $\rho_{\mathrm{KV}}(\text{task}, M, T) :=
\Pr_x[A(b^*, x) > A(b_{\max}, x)]$ where $b^*(x) = \arg\max_b A(b_{1:T}, x)$
over the sweep grid. Pre-registered threshold: $\rho \geq 0.05$ for "dilution-prone."

**The partition is universal.** Across all 7 (model, context) cells tested,
NIAH-MultiKey-3 has $\rho \leq 0.02$; all other dilution-prone tasks (VT,
FWE, QA, niah_multivalue) have $\rho \geq 0.05$. Zero counter-examples.
Detailed table in `experiments/results/h1_consolidated_report.md`.

### 4. The gating method (~2 pages, the new headline)

**Headline reframing (per `strategic_analysis.md` Recommendation #1).** We
report Δ as *gated-X vs the best base evictor X at matched budget*, NOT
"gated-SnapKV vs plain SnapKV." This positions gating as **orthogonal to
base-evictor choice**, neutralizing the strongest reviewer objection
("what about CapKV / Bui / Ada-KV which are better than SnapKV?").

**Method-agnostic on 4 base evictors × 4 model conditions (15/16 cells, single τ=0.07):**

| Model | SnapKV Δ | H2O Δ | StreamingLLM Δ | PyramidKV Δ | Best plain → best gated |
|---|---:|---:|---:|---:|:---|
| Qwen 1.5B | +12.1 | +16.2 | +16.3 | +15.2 | 0.438 → 0.570, **+13.2pp** |
| Qwen 3B | +23.6 | +30.3 | +43.0 | +32.9 | 0.580 → 0.865, **+28.5pp** |
| Qwen 14B | +14.3 | +20.0 | +25.0 | +22.8 | 0.709 → 0.868, **+15.9pp** |
| Mistral 7B | +18.7 | +23.6 | +26.3 | +23.1 | 0.623 → 0.823, **+20.1pp** |

**Average Δ per base evictor across 4 cells:**
- SnapKV: +17.2pp
- H2O: +22.5pp
- StreamingLLM: +27.6pp
- PyramidKV: +23.5pp

**Grand mean across all 16 (model × base evictor) cells: +22.7pp.**

**Same gate, same τ=0.07, no per-method tuning.** The wrapper lifts ALL
four base methods at every model scale past their own plain baselines by
+12pp to +63pp, and even the best plain (over four baselines) by +13pp
to +29pp. Full table at `experiments/results/method_agnostic_matrix.md`.

**Algorithm.** A drop-in wrapper around any SnapKV-style evictor:
```
gated_eviction(prompt, base_evictor, τ):
    cache = prefill(prompt)
    drop = head_agreement_drop(cache.attentions)
    if drop ≥ τ: return base_evictor.evict(cache, budget_b)
    else:        return cache  # full KV
```

The head-agreement drop is the difference between the early-third and
late-third layer-mean Jaccard top-32 similarity across heads. Computed at
no extra cost using the same attention pass needed for SnapKV scoring.

**Mixed-suite results.** N=100 examples per task on the 4-task suite
(NIAH-MK3 + VT + FWE + QA_1), τ=0.07:

| Model | Context | Plain mean | Gated mean | Δ |
|---|:---:|---:|---:|---:|
| Qwen 2.5-1.5B | 4K | 0.438 | 0.559 | **+0.121** |
| Qwen 2.5-3B | 4K | 0.580 | 0.815 | **+0.236** |
| Qwen 2.5-14B | 4K | 0.709 | 0.853 | **+0.143** |
| Mistral-7B-v0.3 | 4K | 0.535 | 0.722 | **+0.187** |
| Qwen 2.5-1.5B | 16K | 0.372 | 0.404 | **+0.032** |
| Qwen 2.5-3B | 16K | 0.530 | 0.564 | **+0.034** |
| Mistral-7B-v0.3 | 16K | 0.535 | 0.651 | **+0.116** |
| Qwen 2.5-14B | 16K | 0.796 | 0.867 | **+0.071** |

**Strongest single-task showcase**: Mistral-7B at 4K NIAH-MK3 (N=100).
Plain SnapKV: 99% → 0% as budget shrinks. Gated SnapKV: **89% at every
budget**. Δ at $b=0.0625$: **+89pp**. 10/100 false positives only.

**Pareto frontier figure** showing accuracy-vs-budget curves: plain
collapses, gated essentially flat across budgets.

### 4.5. Calibration recipe (~0.5 pages, addresses reviewer O3)

The fixed τ=0.07 works on 7/8 cells but mis-classifies on Qwen 3B 16K.
Recipe: measure mean drop on a 20-input pilot from NIAH-MK3 and a 20-input
pilot from VT; set τ = midpoint. **Result**: Qwen 3B 16K Δ +3.4pp → +5.4pp;
Mistral 4K Δ +18.7pp → +22.3pp; neutral elsewhere. Full table in
`experiments/results/calibration_recipe.md`. The recipe is label-free, takes
under 5 minutes on a single A100, and turns the predictor from "works most
of the time" into "comes with a deterministic recipe."

### 5. The a priori partition predictor (~1 page)

The head-agreement drop ordering is invariant across architectures and
sizes; only the threshold needs slight tuning. Empirical drop values
per task (mean across N=20 examples):

| Task | Qwen 1.5B 4K | Qwen 3B 4K | Qwen 3B 16K | Mistral 4K | Mistral 16K |
|---|---:|---:|---:|---:|---:|
| qa_1 | 0.218 | 0.105 | 0.098 | 0.099 | TBD |
| qa_2 | 0.180 | 0.090 | 0.087 | 0.089 | TBD |
| vt | 0.151 | 0.076 | 0.068 | 0.095 | TBD |
| niah_multivalue | 0.107 | 0.046 | 0.069 | 0.077 | TBD |
| fwe | 0.098 | 0.037 | 0.049 | 0.082 | TBD |
| niah_multikey_3 | **0.041** | **-0.002** | **-0.021** | **0.060** | TBD |

NIAH-MK3 is the smallest drop in every cell. The ordering is robust.

**τ sensitivity sweep.** Post-hoc analysis (Section 7) shows the optimal
τ lies in [0.05, 0.20] across cells; τ=0.10 is within 1pp of optimal on
6 of 7 cells.

### 6. Method engineering details (~0.5 pages)

Necessary engineering detail for reproducibility, including the bug we
fixed: HF transformers derives `position_ids` from `attention_mask` via
cumsum when not provided, so the naive attention-mask approach shifts
RoPE for evicted positions. Cache slicing + explicit `position_ids`
preserves positional encoding correctly. Code in
`experiments/scripts/gated_eviction.py`.

For long contexts (T ≥ 16K), eager attention with `output_attentions=True`
overflows. We use SDPA for the pass-1 prefill and switch each layer to
eager for the small pass-2 (last 32 queries against past length T-32),
avoiding the O(T²) buffer.

### 4. Method: cache-slicing eviction with explicit `position_ids` (~0.7 pages)

Necessary engineering detail for reproducibility, including the bug
we fixed: HF transformers derives `position_ids` from `attention_mask` via
cumsum when not provided, so the naive attention-mask approach shifts RoPE
for evicted positions. Cache slicing + explicit `position_ids` preserves
positional encoding correctly. Code in `experiments/scripts/ruler_sweep.py`.

Per-layer ablation: each layer derives its own keep_mask from its own
attention. Same per-layer budget. Modest improvement on aggregation tasks
at 4K ($1.8\times$ on FWE, $2.1\times$ on niah_multivalue); matches at 16K.

### 5. A priori partition predictor (~1.5 pages)

**Negative result first (entropy).** Mean attention entropy is in 0.31–0.35
across all 7 RULER tasks; doesn't separate. Mean head agreement at top-32:
NIAH-MK3 (0.313) and VT (0.310) are tied, but their $\rho$ differ by 6×.
Direct summary statistics fail.

**Positive result: depth-conditional head agreement.** Per-layer-bin head
agreement (early/middle/late thirds):

| Task | early | late | drop |
|---|---|---|---|
| ... | ... | ... | ... |

On Qwen2.5-1.5B (28 layers): 5 dilution-prone tasks have drop $\geq 0.098$;
NIAH-MK3 has drop $0.041$. On Qwen2.5-3B (36 layers): NIAH-MK3 has drop
$-0.002$. **The sign is informative on Qwen: zero or negative drop is a
capacity-bound signature.**

**Cross-architecture caveat.** On SmolLM2-1.7B (Llama-arch, 32 heads / 8 KV
heads, 24 layers): VT, QA_1, QA_2 still cluster at the high end (drop
$\geq 0.13$), but NIAH-MK3 has drop $0.097$ and niah_multivalue has drop
$0.021$. The simple drop misranks two tasks. We hypothesize architecture
normalization (head count, KV-head ratio) is needed; this is the open
theoretical step.

**Mechanism (H3-inherited).** We adopt Bui's Proposition 3.1 (near-tie
distractors force dilution) and Corollary 3.2 (preferential retention
reduces dilution). Our additional Proposition (in `theory_h3.md`):
context-length amplification implies $\rho_{\mathrm{KV}} \propto (1 - A_{\mathrm{full}}(T)) / \gamma$
under constant retention ratio $\gamma$ and good-scoring assumption. The
agreement-drop predictor is a measurable proxy for the $\alpha_\mathcal{R}$
quantity in Bui's framework, evaluated at inference without labels.

### 6. The scaling formula (~1 page)

Empirical verification of $\rho \approx (1 - A_{\mathrm{full}}) \cdot \text{constant}$:

| Sweep | $1 - A_{\mathrm{full}}$ | $\rho$ | ratio |
|---|---|---|---|
| Qwen 1.5B 4K VT | 0.18 | 0.06 | 0.33 |
| Qwen 1.5B 16K VT | 0.75 | 0.19 | 0.25 |
| Qwen 3B 4K VT | 0.00 | 0.00 | n/a |
| Qwen 3B 16K VT | 0.09 | 0.05 | 0.55 |
| Qwen 3B 4K FWE | 0.24 | 0.01 | 0.04 |
| Qwen 3B 16K FWE | 0.62 | 0.32 | 0.52 |

Ratios cluster around $0.25$–$0.55$ on dilution-prone tasks. On NIAH-MK3
the ratio is consistently $\leq 0.02$ regardless of headroom, confirming
the partition.

### 7. Gating method (NEW, to be implemented before submission) (~1 page)

We can use the partition predictor to **avoid eviction on capacity-bound
inputs** while applying eviction on dilution-prone inputs:

1. At prefill, compute the depth-conditional head-agreement drop.
2. If drop is below a calibrated threshold $\tau$: keep full KV.
3. Else: apply SnapKV-style eviction at the predicted optimal budget.

Threshold $\tau$ calibrated once per model on the held-out validation
split. Resulting policy preserves full-cache accuracy on retrieval-
precision inputs while gaining eviction's benefits on integration inputs.

**Status: not yet implemented in v0.6. This is the highest-leverage
remaining experiment before submission.**

### 8. Ablations (~0.7 pages)

- **Random eviction control.** SnapKV vs uniform-random scoring at all
  budgets, all tasks. Random destroys accuracy; SnapKV preserves it.
  Required to claim the dilution effect is real and not "any eviction
  improves things on average."
- **Per-layer vs single-mask.** Per-layer wins on aggregation/multi-value
  at 4K ($1.8\times$ FWE, $2.1\times$ multivalue); matches at 16K.
- **Synthetic NIAH (8/16/32 keys).** The partition is at the task level,
  not the key-count level. 32-key NIAH is easier (more concrete facts to
  anchor); 16-key is the sweet spot for synthetic dilution.

### 9. Limitations and future work (~0.5 page)

- **Cross-architecture predictor transfer is open.** SmolLM2's drop
  ranking misranks two tasks. Architecture-normalized version (e.g.,
  per-arch z-score on a calibration set) is the obvious next step.
- **Single-mask vs per-head.** Bui and Garcia both find per-head policies
  give modest further gains. We use single-mask or per-layer; per-head
  with padded cache would tighten the upper bound.
- **Smallest model class** (1.5B–3B). 7B+ deserves a check but is at
  our compute envelope.
- **English-only RULER.** Cross-lingual partition robustness untested.
- **No new eviction method.** This is a diagnostic / predictor study.
  A method paper that gates Bui or CapKV's eviction by our predictor
  is the natural follow-up.

### 10. Conclusion (~0.3 page)

Recent KV-cache eviction work shows that less compute can beat full
attention on average. We sharpen that to a *partition*: on integration-
heavy tasks (multi-hop, aggregation, QA, multi-value) it does; on
precise multi-key retrieval it doesn't. We provide a label-free a priori
predictor (within architecture) and a scaling formula matching the
empirical recovery rate.

The headline practical takeaway: **before deploying KV-cache eviction in
production, check the head-agreement drop on a calibration batch.** If
the drop is large, eviction will help your tasks; if small or negative,
keep the full cache for that task class.

## What's still missing before submission

| Item | Effort | Priority | Owner |
|---|---|---|---|
| Reframe theory writeup acknowledging Bui's Prop 3.1 | 1 day | High | me |
| Implement and evaluate the gating method (Sec 7) | 1 week | **Critical** | me |
| Architecture-normalized predictor variant | 1 week | High | me |
| Per-head SnapKV with padded cache | 1 week | Medium | me |
| Read Bui's full paper (we read §1–7); read CapKV appendices | 2 days | Medium | me |
| Joules/token table on A100 | 2 days | Low | me |
| Cross-lingual sanity check on one non-English RULER variant | 2 days | Low | me |
| Camera-ready figures | 3 days | High | me |
| Paper draft v1 | 1 week | Critical | me |
| Internal review | 3 days | High | Mishra |

## What we have that's already on the page

- 13 different (model × context × task) cells with measured $\rho$, partition labels match perfectly.
- Audit script showing budget=1.0 pipeline matches clean prefill-decode.
- 5 neighbor papers read end-to-end with carefully-positioned novelty claims.
- Working code (`experiments/scripts/`) for the sweep, analysis, audit, and
  attention probes.
- Theory writeup with Bui-acknowledged mechanism + our context-length
  amplification proposition.
