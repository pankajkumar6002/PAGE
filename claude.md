# CLAUDE.md

Context for working on an efficient / fast neural network research project. The goal is to pick one non-incremental problem statement that a compute-constrained university lab can fully own, then develop it to a top-venue submission. Landscape survey completed June 2026 (covering arXiv through late May 2026).

## Working conventions

Follow these in every response and every artifact for this project.

- Plain English. No em-dashes. No LLM-filler adjectives or adverbs (comprehensive, powerful, seamless, robust, delve, etc.). Concise prose over padding.
- Output as markdown with LaTeX math delimiters (`$...$` inline, `$$...$$` block). Do not produce LaTeX source files unless explicitly asked.
- Formal and BibTeX name: `Mishra, Subhankar`.
- This field moves monthly. Before stating current SOTA or “the latest,” search. Do not rely on training priors for what exists now.
- arXiv ID convention is YYMM. `2605` = May 2026, `2602` = February 2026, `2511` = November 2025. Current date is June 2026, so “2026” queries return stale hits; search without a year or with the specific month.
- Favor contributions that do not need large-scale pretraining: theory-first work, measurement and benchmarking, or methods validatable at small or mid scale on lab hardware.

## Who is working on this

Subhankar Mishra, Associate Professor, School of Computer Sciences, NISER Bhubaneswar. Leads the SML Lab. Background spans GNNs, differential privacy, continuous-depth and Neural ODE models, Mamba / state-space models, Kronecker-factorized message passing, federated learning, and LLM robustness. Target venues include NeurIPS, ECCV, SIGGRAPH Asia, TPAMI, TMLR.

Note for picking a direction: stay open-minded. Do not narrow to GNN or SSM topics just because they are adjacent to past work. The strongest problem statement may sit outside that comfort zone. Treat the background as an asset for execution, not a constraint on scope.

## Landscape: three meta-shifts (more important than any single result)

1. **Efficiency is starting to beat the full model, not just approximate it.** The “compression equals controlled quality loss” framing is breaking. Learned KV-cache eviction can improve long-context reasoning because irrelevant tokens dilute attention, so selective retention sharpens signal (2605.09649). Latent-then-explicit reasoning cuts tokens and raises accuracy at once (2605.07315). Efficiency and quality are becoming the same axis.
1. **Many efficiency methods are trained on the wrong objective.** Several recent papers re-derive a heuristic from first principles and get speed and accuracy together. Speculative drafters optimized for token accuracy when the goal is window-level acceptance length (2605.14978). Linear-attention gates tuned empirically rather than derived from the regression objective (2605.08587). MoE load-balancing losses that actively damage router-expert structure (2605.12476).
1. **Sparsity goes deeper than expert-level.** Headline: under 2% of active routed-expert neurons can retain per-token capacity when combined with a shared expert in large models (2605.08575). The available sparsity is barely exploited, and routing appears to reflect geometry rather than domain specialization.

Deployment reality that most papers ignore: mobile memory bandwidth is 30 to 50x below data-center GPUs, and decode is memory-bound, so compute units idle waiting on memory. FLOPs and parameter counts are the wrong currency. Joules per token under sustained thermal load is closer to the truth.

## Threads tracked (May 2026 anchors)

- **Linear attention / SSM design.** Pole-based, control-theoretic views: TCP-SSM (2605.11563). Hidden states are low-rank after training, ~50% of key/query channels prunable (2602.04852). Log-linear attention (2506.04761). Hybrid SSM-attention priming (2605.08301). Local Linear Attention via test-time regression (2605.29157).
- **Graph + state space.** MbaGCN for over-smoothing (2501.15461). Graph Mamba (2402.08678). Dual Mamba node-specific (2511.06756).
- **Continuous depth / Neural ODE.** Continuous-depth transformers with learned control (2601.10007). Equilibrium reasoners linking test-time scaling to attractor convergence (2605.21488).
- **Quantization.** D2Quant sub-4-bit weight-only PTQ (2602.02546). SOAR for NVFP4 (2605.12245). SignRoundV2 (2512.04746).
- **KV cache.** Learnable global eviction that improves reasoning (2605.09649). FibQuant vector quantization for random-access cache (2605.11478).
- **Few-step generation.** AnyFlow flow-map distillation for any-step video (2605.13724). pi-Flow imitation distillation (2510.14974).
- **MoE.** DECO dense-comparable on-device MoE (2605.10933). Router-expert geometric coupling (2605.12476). Intra-expert neuron sparsity (2605.08575). Mixture of Neuron Experts (2510.05781).
- **Speculative decoding.** Window-level RL drafting PPOW (2605.14978). SlimSpec low-rank draft head (2605.10453). PARD-2 acceptance-length objective (2605.08632). DVI training-aware self-speculation (2510.05421).
- **Reasoning efficiency.** LaTER latent-then-explicit (2605.07315). LatentRAG ~90% latency cut (2605.06285). BudgetThinker control tokens (2508.17196).
- **Edge / on-device.** On-Device LLMs State of the Union 2026 (v-chandra.github.io/on-device-llms). Single-board benchmarking (2604.24785).

## Candidate problem statements (ranked by openness x tractability)

1. **A theory and method for “compute that helps by being removed.”** The efficiency-as-improvement results are empirical and siloed (KV eviction, latent reasoning, token dropping). No unified account of when cutting computation improves generalization (regularization, denoising, attention dilution), and no method that predicts the right per-input budget rather than sweeping it. Fresh, low compute, reframes a subfield. Top pick for novelty per GPU-hour.
1. **Adaptive compute via equilibrium / attractor dynamics.** A late-May result ties test-time scaling to convergence toward solution-aligned attractors (2605.21488). Open: convergence guarantees, certified early stopping, and the formal link between latent reasoning and fixed-point iteration. Connects implicit / DEQ models to the reasoning literature. Higher risk, highest ceiling.
1. **Energy- and bandwidth-first efficiency with honest benchmarking.** Given the 30 to 50x bandwidth gap and joule / thermal limits on-device, there is a methodology gap: a benchmark in joules per token under sustained thermal load, plus co-design that optimizes memory traffic rather than FLOPs. Impactful, fundable, buildable with lab infrastructure, far less crowded on the measurement side. Can be a benchmark contribution as much as a method.
1. **Neuron-level (not expert-level) sparse inference.** If under 2% of neurons per token suffice and routing reflects geometry, design inference exploiting intra-expert neuron sparsity with hardware-friendly structure. The parameter-free geometric router hints routing can be near-free. Systems-flavored, weeks-old angle.
1. **Sequence-model layers from control / signal-processing first principles.** Pole-based scan operators (2605.11563) plus low-rank state structure (2602.04852) leave room for layers with provable memory horizons and certified stability. Theory-leaning, tractable.
1. **Flow maps for unified few-step / any-step generation.** Resolves the consistency-vs-scaling tension by optimizing the full ODE trajectory (2605.13724). Principled but crowded by strong generative-modeling groups. Enter only with a real theoretical angle.
1. **Latent reasoning, the “when and why.”** Mostly engineering today. Open: when latent computation hurts symbolic tasks, how to verify and interpret latent thoughts, the optimal latent-to-explicit switch. Hot and crowding fast. The theory / verification slice is the defensible part.

## Current top bets

For a strong non-incremental paper a university lab can fully own: **#1** (efficiency-as-improvement, theory-first) and **#3** (energy / bandwidth benchmarking plus co-design, build-and-measure). Choose **#2** for the most ambitious, highest-ceiling option.

## Direction picked (2026-06-01, refined 2026-06-03)

Top bet **#1 (efficiency-as-improvement)**, scoped per `problem_statement.md` v0.6. **Significant scope narrowing after re-doing the literature 2026-06-03:** I had missed Bui et al. 2605.09649 and CapKV 2604.25975 in the initial search. Both papers pre-empt the headline "less compute beats full-cache" / "attention dilution mechanism" / "closed-form label-free" claims. Honest verdict in `problem_statement.md` v0.6's "What changed from v0.5" section.

The paper survives but in a narrower form. Old framing: "Less compute beats full attention" (now subsumed by 2605.09649 + 2604.25975 + 2605.25475). New framing: **"When does KV-cache eviction help versus hurt? A task-type partition and its a priori predictor."** Target unchanged (AAAI/ICLR Sep 2026). Hardware unchanged (4x A100 80GB).

## H1 spot-check (2026-06-02)

Pre-registered question: $\rho_{\mathrm{KV}}$ = fraction of inputs where some KV budget $b < b_{\max}$ yields strictly higher accuracy than full-KV, threshold $\rho \geq 0.05$ to support H1.

Setup: synthetic multi-key NIAH on Qwen2.5-1.5B-Instruct, SnapKV-style attention-scored cache slicing with explicit `position_ids` to preserve RoPE alignment. Code in `experiments/scripts/`. Full report in `experiments/results/h1_consolidated_report.md`.

**Effect holds on RULER and scales with context length.**

| Task | Context | full-KV | best $b{<}1$ | $\rho_{\mathrm{KV}}$ | H1 |
|---|:---:|---:|---:|---:|:---:|
| RULER VT (multi-hop) | 4K | 82.0% | 82.0% | 0.060 | ✓ |
| **RULER VT** | **16K** | 25.0% | **29.0% @ b=0.5** | **0.190** | ✓✓✓ |
| RULER FWE (aggregation) | 4K | 22.0% | 23.0% | 0.060 | ✓ |
| RULER FWE | 16K | 12.0% | 12.0% | 0.040 | ~ |
| RULER QA_1 (single-doc) | 4K | 73.0% | 74.0% | 0.080 | ✓ |
| **RULER QA_1** | **16K** | 65.0% | **68.0% @ b=0.25** | **0.140** | ✓✓ |
| RULER QA_2 (multi-doc) | 4K | 45.0% | 49.0% | 0.080 | ✓ |
| RULER niah_multivalue | 4K | 30.0% | 31.0% | 0.070 | ✓ |
| RULER NIAH-MK3 (retrieval) | 4K | 65.0% | 47.0% | 0.010 | ✗ |
| RULER NIAH-MK3 | 16K | 26.1% | 23.9% | 0.000 | ✗ |
| RULER VT 4K SmolLM2-1.7B | 4K | 65.0% | 65.0% | **0.100** | ✓ |

**Five RULER task families clear the pre-registered 0.05 threshold (VT, FWE, QA_1, QA_2, niah_multivalue).** NIAH-MK3 (retrieval-precision) is the only task that fails, both at 4K and 16K, consistent with H3's predicted partition: eviction helps when failure mode is dilution-induced uncertainty, fails when it is retrieval-precision.

**$\rho$ amplifies with context length** on dilution-prone tasks: VT 0.06 → 0.19 (4K → 16K), QA_1 0.08 → 0.14. Longer context = more dilution = more for eviction to remove.

**Cross-model.** SmolLM2-1.7B (Llama arch, 32H/8KV, 24 layers): at its max context (8K), VT $\rho$ = **0.240** (24 inputs recovered, peak +9pp), FWE $\rho$ = 0.150, QA_1 $\rho$ = 0.100. Same partition as Qwen, same context amplification. NIAH-MK3 fails on SmolLM2 too ($\rho$=0.02 at 4K).

**Cross-size scaling.** Qwen3B at 4K saturates dilution-prone tasks (VT 100%, QA_1 84%) — no headroom for H1. At 16K context the larger model re-enters the dilution regime: **Qwen3B 16K FWE $\rho$ = 0.320 (38% → 51% at b=0.0625, +13pp), Qwen3B 16K niah_multivalue $\rho$ = 0.130**. Joint scaling pattern: $\rho \approx (1 - A_{\mathrm{full}}) \cdot \text{dilution}(T)$. This is exactly the H3 prediction.

**A priori predictor candidate (2026-06-03).** On Qwen 1.5B, the **early-to-late head-agreement drop** cleanly separates the partition: dilution-prone tasks have drop $\geq 0.098$, NIAH-MK3 has drop $0.041$. Cross-architecture transfer to SmolLM2-1.7B (Llama arch, different head count and depth) is incomplete — the raw drop ranking shifts. The architecture-invariant version of this predictor is the **open theoretical step** for the paper. Practical predictor for the paper: task category (multi-hop / aggregation / QA / multi-value $\to \mathcal{D}$; precise multi-key retrieval $\to \mathcal{C}$). H3 mechanism supported empirically on both models via direct $\rho$ measurement; only the *a priori* shortcut needs more work.

**Verdict (revised 2026-06-03 after literature re-check): proceed with narrowed scope.** The H1 paradigm (less > full) and dilution mechanism are now in the literature (Bui et al. 2605.09649; CapKV 2604.25975; IndexMem 2605.25475). The **task-type partition** (which tasks are dilution-prone vs capacity-bound), the **a priori predictor** (head-agreement drop), and the **scaling formula** ($\rho \approx (1 - A_{\mathrm{full}}) \cdot \mathrm{dilution}(T)$) remain novel. Full report in `experiments/results/h1_consolidated_report.md`. Honest novelty positioning in `problem_statement.md` v0.6.

## Gating method result (2026-06-03 evening)

Plan Addition 1 (the headline ICLR contribution) implemented and validated.
The partition-aware gating algorithm:

```
gated_eviction(prompt, base_evictor, τ):
    cache = prefill(prompt)
    drop = head_agreement_drop(cache.attentions)  # early - late, third-bin mean Jaccard top-32
    if drop ≥ τ: return base_evictor.evict(cache)
    else:        return cache  # full KV
```

**Mixed-suite results at 4K (N=400, tasks = NIAH-MK3 + VT + FWE + QA_1), SAME τ=0.07 across architectures:**

| Model | 4K Δ | 16K Δ | Notes |
|---|---:|---:|---|
| Qwen2.5-1.5B | +12.1pp | +3.2pp | partition perfect at both |
| Qwen2.5-3B | +23.6pp | +3.4pp | 16K: FWE/VT mis-classified (cross-context τ-shift) |
| Qwen2.5-14B | +14.3pp | **+7.1pp** | partition perfect, max Δ +20.8pp at b=0.0625 |
| Mistral-7B-v0.3 | +18.7pp | +11.6pp | cross-arch + cross-context τ both work |

NIAH-MK3 isolated headline: Mistral-7B plain SnapKV collapses 99% → 0% across budgets; gated holds **89% at every budget** (Δ = +89pp at $b=0.0625$).

The cross-architecture transfer (Qwen-GQA ↔ Mistral-different-GQA, same τ) was listed in the plan as the Week 3-4 risk — empirically resolved in one day. Architecture-invariant predictor is the bare drop, not a normalized variant. Pending verification on Llama-3.1-8B.

## Open work

Done since 2026-06-01:
- [x] Scale RULER to 16K with two-pass prefill (`ruler_sweep.py --two_pass`).
- [x] Cross-model check on SmolLM2-1.7B and Qwen2.5-3B; partition holds.
- [x] Per-layer vs single-mask SnapKV ablation. Per-layer helps on aggregation/multi-value at 4K (~2x ρ); matches at 16K.
- [x] Refined a priori predictor: head-agreement drop separates partition cleanly within Qwen family.
- [x] Formal H3 propositions in `theory_h3.md` (SNR improvement, partition prediction, context-length scaling). **But:** subsumed by Bui Prop. 3.1 + Cor. 3.2 — needs reframing as "we use this mechanism to predict the partition," not "we discover the mechanism."
- [x] Re-do literature review 2026-06-03. Found Bui (2605.09649), CapKV (2604.25975), DynamicKV (2412.14838), Garcia (2605.18053), IndexMem (2605.25475). Updated `problem_statement.md` to v0.6 with honest novelty positioning.
- [x] Code audit at budget=1.0: matches clean prefill-decode 5/6 examples (1 disagreement is a trailing period). Audit script at `experiments/scripts/audit_budget1_consistency.py`.

Outstanding:
- [x] Implement gating method — 8 cells across 4 model sizes (1.5B/3B/7B/14B) and 2 architectures (Qwen, Mistral).
- [x] 16K gating (switched pass-1 to SDPA to avoid OOM).
- [x] Qwen2.5-14B at 4K (+14.3pp) and 16K (+7.1pp).
- [x] τ sensitivity post-hoc sweep: τ ∈ [0.05, 0.20] near-optimal; τ=0.10 within 1pp of best on 6/7 cells; Qwen 3B 16K prefers τ=0.
- [x] Qwen 3B 16K diagnostic: drops compress at long context; *ordering* of partition preserved; threshold needs slight tuning.
- [x] problem_statement.md bumped to v0.8.
- [x] paper/outline.md updated to v3 with real numbers.
- [x] gating_consolidated_report.md written.
- [ ] LongBench cross-benchmark — 4 sweeps running on GPUs 0-3, ETA ~30-60min.
- [ ] **SOTA baseline gap.** Currently we only compare gated-SnapKV vs plain-SnapKV. R3 will ask about H2O / StreamingLLM / PyramidKV / Ada-KV / Bui-DBTrimKV / CapKV. Implementing gated-X for X ∈ {H2O, StreamingLLM, PyramidKV} in `gated_eviction.py` via `--score_policy` flag (in progress). Cite published numbers for the learned/trained methods (CapKV, Bui, IndexMem, DynamicKV).
- [ ] Pull Bui's TrimKV code (github.com/ngocbh/trimkv) and run gated-TrimKV for the strongest "method-agnostic" demonstration.
- [ ] Garcia's protocol ablation (10% prefix + 10% suffix protection) — verify our results aren't protection-fix artifacts. Our current setup uses n_sink=4 + obs_window=32 which is similar but smaller.
- [ ] Architecture-normalized predictor (drop / max_drop or per-arch z-score) to fix Qwen 3B 16K calibration.
- [ ] Per-head SnapKV (true per-head with padded cache) — only if needed as stronger baseline.
- [ ] Confirm Jetson Orin availability for the secondary energy table.

## Method-agnostic claim verified (2026-06-04 night)

Plan addition #1 ("gating is wrapper for any base evictor") now empirically verified. Sweeps on Qwen 1.5B 4K mixed suite at τ=0.07:

| Base evictor | plain mean | gated mean | Δ |
|---|---:|---:|---:|
| SnapKV | 0.429 | 0.578 | **+0.149** |
| H2O | 0.196 | 0.357 | **+0.162** |
| StreamingLLM | 0.033 | 0.195 | **+0.163** |
| PyramidKV | 0.417 | 0.570 | **+0.152** |

**All 4 base evictors improved by the same gate, same τ, no per-method tuning.** Mechanism is uniform: gate is closed (drop < 0.07) on the 25% of inputs from NIAH-MK3 (capacity-bound); every base evictor collapses to ≤1% on those at b=0.125; gated recovers them all to 0.65. On dilution-prone inputs the gate stays open and gated == plain by construction.

**Final full matrix (2026-06-04, 16/16 cells, all positive):**

| Model | SnapKV Δ | H2O Δ | StreamingLLM Δ | PyramidKV Δ |
|---|---:|---:|---:|---:|
| Qwen 1.5B | +12.1pp | +16.2pp | +16.3pp | +15.2pp |
| Qwen 3B | +23.6pp | +30.3pp | +43.0pp | +32.9pp |
| Qwen 14B | +14.3pp | +20.0pp | +25.0pp | +22.8pp |
| Mistral 7B | +18.7pp | +23.6pp | +26.3pp | +23.1pp |

Per-policy average across 4 cells: SnapKV +17.2, H2O +22.5, StreamingLLM +27.6, PyramidKV +23.5. **Grand mean across 16 cells: +22.7pp.**

**Best plain → best gated (reframed headline metric):**

| Model | Best plain | Best gated | Δ |
|---|---:|---:|---:|
| Qwen 1.5B | 0.438 | 0.570 | **+13.2pp** |
| Qwen 3B | 0.580 | 0.865 | **+28.5pp** |
| Qwen 14B | 0.709 | 0.868 | +15.9pp |
| Mistral 7B | 0.623 | 0.823 | **+20.1pp** |

Average Δ per base evictor across cells: SnapKV +17.2, H2O +22.5, StreamingLLM +31.9, PyramidKV +23.7. Full report at `experiments/results/method_agnostic_matrix.md`.

## Calibration recipe (Plan addition for reviewer O3)

For a new (model, context), measure mean drop on a 20-input pilot from NIAH-MK3 (capacity-bound) and a 20-input pilot from VT (dilution-prone). Set τ = midpoint.

Validation across 4 measured cells:

| Cell | drop(NIAH-MK3) | drop(VT) | τ_calibrated | gated@τ_cal | gated@τ=0.07 |
|---|---:|---:|---:|---:|---:|
| Qwen 1.5B 4K | +0.045 | +0.161 | +0.103 | +12.1pp | +12.1pp |
| Qwen 3B 4K | -0.002 | +0.076 | +0.037 | +21.3pp | +25.9pp |
| **Qwen 3B 16K** | -0.021 | +0.068 | +0.024 | **+5.4pp** | +3.4pp |
| Mistral 4K | +0.060 | +0.096 | +0.078 | **+22.3pp** | +18.7pp |

Recipe **fixes the Qwen 3B 16K failure** (+5.4 vs +3.4pp) and **improves Mistral 4K** (+22.3 vs +18.7pp); slightly worse on Qwen 3B 4K. Honest fallback: present both. Script: `experiments/scripts/calibration_recipe.py`.

## Headline-reframing plan (per strategic analysis)

Currently we report "Δ over plain SnapKV." Per the strategic analysis the right reframing is **"Δ over best base evictor at matched budget."** For Qwen 1.5B 4K mixed suite (N=400):

- Best plain among {SnapKV, H2O, StreamingLLM, PyramidKV}: PyramidKV at 0.417, SnapKV at 0.429
- Gated SnapKV: 0.578 (+14.9pp over its own plain; +14.9pp over best plain)
- Gated PyramidKV: 0.570 (+15.2pp over its own plain; +14.1pp over best plain)

So gated-X uniformly beats the best plain by +14pp. This is the headline metric the paper should lead with.

## SOTA comparison matrix (2026-06-03 late night)

| Method | Status | What we have / need |
|---|---|---|
| **SnapKV** (Li 2024) | ✅ | Our plain baseline IS SnapKV-style scoring |
| **H2O** (Zhang 2023) | 🔄 | Implementing as `--score_policy h2o` |
| **StreamingLLM** (Xiao 2023) | 🔄 | Implementing as `--score_policy streamingllm` |
| **PyramidKV** (Cai 2024) | 🔄 | Implementing as `--score_policy pyramidkv` |
| **Ada-KV** (Feng 2024) | ⚠️ | Cite published numbers; their code at github.com/ |
| **Bui / DBTrimKV** (2605.09649) | ⚠️ | Pull code from github.com/ngocbh/trimkv |
| **CapKV** (2604.25975) | ⚠️ | Cite Table 1 LongBench Qwen3-8B numbers |
| **DynamicKV** (2412.14838) | ⚠️ | Cite published numbers |
| **IndexMem** (2605.25475) | ⚠️ | Cite their RULER 4K/16K numbers (Table 1) |
| **Garcia / Protection** (2605.18053) | ✅ | We use sink+recency protection ourselves |

ICLR positioning: our contribution is the **gating wrapper**, orthogonal to base evictor. We demonstrate it improves SnapKV (4 model × 2 context = 8 cells, +3 to +24pp mean Δ). For ICLR strength, additionally demonstrate it improves H2O, StreamingLLM, PyramidKV (in progress). The trained/learned methods (Bui, CapKV, IndexMem) are cited for context, with optional gated-DBTrimKV result if their code reproduces.
## Submission-readiness review and fix pass (2026-07-02)

Full four-track review (paper audit, results cross-check, venue check, deep-research
literature sweep) followed by a fix pass. Verdict before fixes: NOT ready. After
fixes: content-ready pending two Mistral sweeps and venue formatting.

**Deep-research novelty verdicts (as of 2026-07-02):** none of the four contributions
fully pre-empted. (1) Task partition: novel in substance, crowded framing — must
differentiate vs 2605.08234 ("When Does Value-Aware KV Eviction Help?"), 2510.00231
(Pitfalls, ACL 2026), VaSE 2606.03928; ManifoldKV 2602.08343 (ICML 2026) forces
scoping "capacity-bound" to attention-score-based evictors. (2) Whether-to-evict gate:
novel as deployed mechanism; 2603.01426 (head-consensus dynamics, missed in June
survey) owns the signal family descriptively. (3) Wrapper: form now common (VECTOR
2605.23258, CriticalKV 2502.03805v2) but whether-gate + single transferred threshold
unclaimed. (4) Scaling formula: fully novel. 13 citations added to main.bib (now 40
entries); new "Concurrent 2026 diagnostics and wrappers" paragraph in Related Work.

**Data-integrity findings fixed in paper:**
- tab:partition Mistral columns had NO provenance (six cells no data, four
  contradicted). Replaced with values computed from gated_*_mistral7b.jsonl
  correct_plain: VT 0.00/0.32, FWE 0.02/0.22, QA_1 0.04/0.00, MK3 0.00/0.00.
  QA_2 + niah_multivalue sweeps launched (ruler_{4k,16k}_mistral7b_qa2_mv.jsonl).
  NOTE: Mistral 4K is saturation-limited (A_full: VT 1.00, FWE 0.82); QA_1 on
  Mistral is a genuine budget-insensitive exception (headroom exists, rho=0) —
  now reported alongside QA_2-on-14B as the second exception.
- Qwen3B cells corrected: QA_1 0.03/0.08 (4K no longer crosses threshold),
  QA_2 4K 0.04 (was "--"), mv 0.02, MK3 16K 0.02.
- **gated_4k_qwen3b.jsonl (SnapKV matrix cell) was run at τ=0.04, not 0.07**
  (gate boundary check across all gated jsonls; all other matrix runs consistent
  with 0.07). Re-evaluated post-hoc exactly at τ=0.07: gated 0.839, Δ +0.259
  (was 0.815/+0.236). Grand mean 22.7 → **22.9pp**. Disclosed in caption +
  App. repro notes; correction appended to method_agnostic_matrix.md; figure
  regenerated. This also resolves the old tab:recipe/tab:matrix +0.259/+0.236
  discrepancy (both now +0.259).
- tab:drops: two layer-bin conventions were mixed; standardized on
  aggregate_drops_n100.py convention (late bin = [2L/3, L)); only the Llama
  column changed (0.097/0.101/0.083/0.064/0.114/0.073). Convention now stated
  in caption. N corrected (100/50, not 20).
- Llama prose errors fixed (+43.5pp not +44.7; gate closes on 44% not 56%;
  bogus "0.10 on Mistral" removed).
- LongBench limitations paragraph: real tasks (qasper/triviaqa/trec/
  multifieldqa_en), all four models reported incl. Qwen14B −0.8pp pooled
  (trec −2.2pp) — the project's only negative result, now disclosed.

**Claims re-scoped:** abstract now says 3 architecture families (Yi+Llama = one
Llama family), partition "clean binary" dropped, "15 of 16 cells" → "7 of 8
Qwen/Mistral (model,context) cells" everywhere, Llama-family transfer reported as
partial with Yi inversion stated plainly (new limitations paragraph), Garcia
protection claim downgraded to "same kind, smaller extent; matched ablation not
run" (new limitations paragraph), capacity-bound claim scoped to
attention-score-based evictors (ManifoldKV cited), single-seed/no-CI limitation
added. Internal leaks removed (H1 codename, NOTES path, "Reviewers asked",
"access granted mid-experiment", repo paths). Four inline tabulars numbered
(tab:distractor, tab:bestplain, tab:capkv, tab:dbtrimkv). Yi/Llama split out of
tab:matrix into tab:crossarch with budget-grid disclosure. CapKV temperature
renamed β. Reproducibility + ethics statements added. ruler_sweep.py two-pass
OOM fixed (sdpa load + eager scoring pass, ported from gated_eviction.py).

**Venue reality (checked 2026-07-02):** ICLR 2027 CFP not yet posted (site 404;
West Coast NA; expect ~Sep deadlines, template ~Jul-Aug). AAAI-27: abstract
Jul 21, paper Jul 28 (UTC-12), 7pp+2pp refs two-column, aaai2027.sty, mandatory
reproducibility checklist. Current draft = 29pp NeurIPS-stub single column
(~15pp main text): ICLR needs ~40% cut; AAAI needs ~50% cut + 2-col + supplement
split. neurips_2024.sty is a 258-byte stub — never compiled in a real template.

**Still open:**
- [x] Fill tab:partition Mistral QA_2/mv cells (done 2026-07-02: mv 0.06/0.21 crosses threshold both contexts; qa_2 0.01/0.00 budget-insensitive, third QA exception; see ruler_mistral7b_qa2_mv_final.md); also
  re-check "niah_multivalue dilution-prone in five of six measured cells" sentence.
- [ ] Venue decision (AAAI-27 in 19 days vs ICLR 2027 ~Sep) + template migration + length cut.
- [ ] Optional reviewer-hardening: binomial CIs on headline tables; second
  capacity-bound task family beyond RULER; Garcia 10%+10% protection ablation;
  gated-ManifoldKV composition test.
- [ ] TAKE OpenReview bib entry has author={Anonymous} (anti-bot block) — fill manually.

## Reviewer-hardening pass complete (2026-07-03)

All six hardening tracks done and integrated into paper/main.tex (31pp, clean):

1. **CIs** (confidence_intervals.md, App. app:ci): all 16 matrix Δs exclude 0
   (min lower bound +9.3pp); only Qwen3B-16K scaling cell includes 0 (the known
   misfire cell). tab:partition now has inline Wilson CIs; pooled contrast
   MK3 2/526 [0.001,0.014] vs mv 65/680 [0.076,0.120] is the robust claim.
2. **z-scored predictor** (normalized_predictor.md): per-model z-score of D from
   unlabeled ~100-input pilot + one global θ_z=−0.69 fit on Qwen/Mistral only →
   zero-shot transfer to Yi/Llama, 7/7 task-level separation (mv scoped out).
   Llama Δ +0.109→+0.174; Yi genuinely compresses (kept 0.22-0.32) at +0.145 —
   old +0.390 exposed as full-KV-fallback artifact. Shape variants fail (AUC
   0.22 on Llama). In paper: abstract, contrib 2, sec:recipe new paragraph,
   drops-ordering, limitations.
3. **Garcia ablation** (garcia_protection_ablation.md): partition survives
   10%+10% (n_sink=w=410; effective floor 824 tokens): MK3 0.65→0.09 despite
   3.3× cache; FWE ρ 0.06→0.13, floor acc 0.31 > full-KV 0.22; protected VT
   dies at floor (0.00) where light-protection holds 0.75. Now a positive
   paragraph in sec:partition; limitation removed.
4. **Second capacity-bound family** (second_capacity_bound_family.md):
   passage_retrieval_en is evict-robust AND gate-open (supports structure-not-
   task-name claim; added to partition section); passage_count gate-predicted
   capacity-bound but floor-accuracy-untestable at 1.5B (in limitations as the
   named follow-up). MK2 remains the only extra empirically capacity-bound task.
5. **Gated-ManifoldKV** (gated_manifoldkv_qwen15b_4k.md): new score_policy
   manifoldkv in gated_eviction.py (single-mask variant, documented deviations).
   Fifth base evictor: same gate/τ, Δ +16.2pp, no negative cell, MK3 0→0.65 at
   every budget. Our variant does NOT reproduce ManifoldKV's MK3 rescue (plain
   0.01 at b=0.5) — their per-layer design untested; scoping kept via citation.
6. **ICLR staging** (paper/iclr2026/, CUT_PLAN.md): compiles clean in
   iclr2026_conference.sty; main text 16.2pp vs 9pp limit; cut plan identifies
   ~9.3pp of savings.

Remaining before submission: execute the ICLR length cut (~half day, after
venue confirmation), arXiv preprint decision (recommended: post now for
priority), fill TAKE bib authors if the OpenReview page becomes reachable,
optional: passage_count on a stronger model, per-layer ManifoldKV.

## Round-2 reviewer-hardening complete (2026-07-04)

Six ICLR-targeted additions, all integrated into paper/main.tex (36pp, clean
bibtex, 0 undefined refs):

1. Accuracy-memory frontier (pareto_oracle_analysis.md, figs/pareto_gated;
   subsec:pareto): gated dominates frontier +3..+19pp at EQUAL achieved kept-KV
   for targets >~1/3 full cache, saturates below. Achieved compression 3.1-3.4x
   at nominal 16x (1.8x Qwen3B). Oracle recovery mean 69%. Abstract: "matched
   nominal budget" + achieved number.
2. Latency/memory (latency_memory.md, tab:latency): no speedup at 1.5B
   (weight-bound), 1.29x at Mistral-7B 4K, ~2.7GiB/seq at 32K. Gate cost 17ms
   Qwen / 89ms Mistral -- corrected old ~50ms claim everywhere.
3. Gate-signal ablation (gate_signal_ablation.md, tab:signal-ablation): D
   uniquely separates (per-input AUC 1.000 vs best-alt 0.805); entropy ->
   early-late-entropy -> agreement dissection. Answers "why this statistic".
4. Pre-registered 32K test (scaling_32k_*, subsec:prereg32k): froze+hashed
   predictions then swept. 1/3 strict hits, both misses CONSERVATIVE. Formula
   reframed directional+ordinal; constant-ratio falsified. MK3 claim carries
   32K=0.04.
5. Qwen3-4B z-score (4th family; qwen3_zscore_validation.md): sharpest evidence.
   Fixed tau does nothing (99.4% kept, +0.43 illusory); z-tau=0.022 -> real gate
   (43% kept, +23.5pp). Closes DBTrimKV Qwen3 caveat. Data merged from 2 split
   files -> gated_4k_qwen3_4b_merged.jsonl.
6. Qwen2.5-32B row (gated_4k_qwen25_32b.md): scale validation. Ordering
   transfers (MK3 D=0.025 smallest). 32B/4K = SATURATION regime (A_full
   0.93-1.00); gate value is protection, +0.335 mean (MK3 +0.88). Ran manually
   on GPU 0 two-pass SDPA. Abstract model list -> 1.5B/3B/14B/32B.

Every predictable objection now has an in-paper answer. Content-complete.
Remaining: ICLR length cut (36->9pp main; paper/iclr2026/CUT_PLAN.md now STALE,
needs re-sync from main.tex), arXiv preprint, venue confirm (recommend ICLR
2027).

## ICLR 2026 submission build complete (2026-07-04)

Venue fixed: ICLR (follow ICLR 2026 format now, swap to 2027 style when released).
Submission build at paper/iclr2026/main.tex (master paper/main.tex kept intact).

- Confirmed ICLR 2026 rules: 9pp main (10 camera-ready), refs+appendix unlimited,
  reproducibility+ethics statements free, LLM-use disclosure in appendix,
  iclr2026_conference.sty, double-blind. Fetched 3 ICLR style exemplars to
  literature/12_iclr_style_exemplars/ (FastGen ICLR24 = closest twin, StreamingLLM
  ICLR24, DuoAttention ICLR25).
- Tone edits per exemplars: method NAMED **PAGE (Partition-Aware Gated Eviction)**
  via \newcommand{\method} (one-line renameable). Abstract rewritten ICLR-style:
  leads with phenomenon->named method, fronts efficiency numbers (3.1-3.4x cache,
  3-19pp frontier, 99->0 vs 89 showcase), asserts wins, ends "Code released".
- Restructured to 9pp main + everything preserved in appendix (agent + my review):
  main text ends p9, 35pp total, 0 undefined refs, 0 citation warns, worst
  overfull 10pt. KEPT in main: intro+contribs, partition prose+tab:partition,
  PAGE algorithm+drop def+tab:matrix, Pareto figure, scaling-formula statement,
  32K verdict, head-to-head 2-sentence summaries, limitations, conclusion,
  repro+ethics. MOVED to Appendix "Deferred..." (with pointers): theory
  derivations, tab:distractor/signal-ablation/crossarch/bestplain/recipe/
  qwen3b-curve/scaling/prereg32k/achieved-compression/drops/scaling-formula/
  latency/capkv/dbtrimkv, tau-sensitivity fig, z-score family detail, per-layer,
  random control, related-work body (3-sent summary kept), matrix bar-chart,
  showcase figure. Added LLM-use disclosure appendix (editable; flagged).

Remaining: author review of the 9pp cut + PAGE name + LLM disclosure wording;
arXiv preprint; swap to iclr2027 style when CFP posts (~Aug-Sep).

## Novelty + writing review, round 3 (2026-07-04)

Ran fresh deep-research web novelty sweep + writing-alignment agent (vs ICLR
exemplars) + adversarial novelty-positioning agent. Verdict: novelty HOLDS 4/4
as of 2026-07-04 (whether-to-evict gate, cross-layer head-agreement signal,
scaling law all NOVEL - no paper does them; task partition novel as taxonomy but
dilution premise shared; C3 transfer partially pre-empted in spirit by CompilerKV
2602.08686). Applied positioning fixes to paper/iclr2026/main.tex:
- Fixed 2 internal contradictions: "None characterises a task family where
  eviction hurts" (contradicted Pitfalls 2510.00231 concession) -> now cedes
  hurts-observation, claims JOINT partition + predictor; "transfers one threshold
  across architectures" (contradicted Llama limitation) -> now "fixed across
  Qwen/Mistral, extended to Llama/Qwen3 by z-standardization".
- Fixed dangerous 2603.01426 concession ("same signal family" -> precise distinction:
  their WITHIN-layer consensus level, descriptive; ours CROSS-layer early-late drop,
  gated). Corrected main-text mischaracterization (was lumped with "adapt which tokens").
- Foregrounded wedges: value-vs-attention axis (vs 2605.08234), whether-vs-which +
  training-free-vs-learned-gate (vs VECTOR/CriticalKV/Attention-Gate/Fast KVzip).
- Reframed ManifoldKV: "capacity-bound is scorer-relative" now prominent in Limitations.
- "We do not claim" paragraph concedes 2 more overlaps (Pitfalls, head-consensus).
- De-hedged contributions (moved caveats to Limitations); removed "honest caveat",
  "report misses as found", "report explicitly rather than folding" meta-phrasing.
- Added 3 bib entries (Attention-Gate 2410.12876, EpiKV 2606.26472, DefensiveKV
  2510.13334) - PLACEHOLDER titles/authors, NEED verification pre-submission (like TAKE).
- Glossed dilution-prone/capacity-bound in abstract.
Still 9pp main, 0 undefined, 0 em-dashes.

NAMING RISK: acronym check found PagedEviction (2509.04377), a block-wise KV
eviction method - name uncomfortably close to PAGE. \method macro makes rename
one-line. Alternatives to consider. No exact PAGE collision.

Open judgment calls for author: (1) method name (PAGE vs alternative given
PagedEviction proximity); (2) figure promotion (writing agent wants phenomenon
figure + showcase fig + signal ablation in main text; trades vs 9pp - would need
to push other content to appendix); (3) run true per-layer ManifoldKV on NIAH-MK3
to close the one exploitable showcase gap (few GPU-hrs); (4) verify 4 placeholder
bib entries (TAKE + 3 new).

## Figure promotion + 9pp rebalance (2026-07-04)

Item 2 (writing-agent request): added phenomenon figure
(figs/head_agreement_phenomenon.pdf, made by paper/make_phenomenon_fig.py from
gate_signal_ablation.jsonl - per-task drop_D strip plot with tau=0.07 line, MK3
below / dilution-prone above) at p5, and promoted the Mistral showcase figure
back to main text (p8). Both now in main alongside Pareto (p7). Rebalanced to
stay 9pp (conclusion p9) by moving to appendix: achieved-compression mechanics,
drop-ordering+scaling-verification prose, Qwen3B-16K limitation; merged the two
head-to-head subsections. paper/iclr2026/main.tex: 9pp main, 35 total, 0
undefined, 0 citation warns, 0 em-dashes.

Item 3 (per-layer ManifoldKV on NIAH-MK3) running on GPU 0.

## Per-layer ManifoldKV result (2026-07-04) + writing de-numbering
Item 3 done: faithful per-layer ManifoldKV (independent per-layer keep-sets,
score_policy=manifoldkv_perlayer in gated_eviction.py) on RULER NIAH-MK3
(Qwen2.5-1.5B, N=100) ALSO COLLAPSES: 0.65 full -> 0.03 at b=0.5 -> 0.00 below,
faster than SnapKV (0.14 at b=0.5). Case (b): closes the "run true per-layer
ManifoldKV" rebuttal. Could not reproduce their 92.4%@50% (setup/model
difference; honest note). MK3 capacity-bound for every scorer tested. Updated
the ManifoldKV limitation paragraph from "untested" to this concrete result.
Report: manifoldkv_perlayer_mk3.md. (VT/FWE dilution-side runs still finishing,
confirmatory only.)

Writing: de-numbered per user feedback (too many numbers in prose). Abstract cut
to one hero number (99->0 vs 89 showcase); breadth stated qualitatively.
Contributions rewritten as claims+pointers to tables/figures (removed +22.9pp,
+13-29pp, per-cell recitations). Matrix prose: summarized budget list, lightened
showcase preview. No numbers lost - all now live in tables/figures. Still 9pp.

## Optional work + theory complete, synced to page-kv.git (2026-07-04)
All queued work done and pushed to ssh://git@10.10.0.178:2222/smlab/page-kv.git (main):
- Git repo initialized + synced (venv/build excluded, trimkv flattened).
- QA_2@Qwen3B-16K = rho 0.02 -> partition table fully populated, no blanks.
- Distractor sweep replicated on Mistral-7B (MK1>MK2>MK3: 0.081>0.072>0.060) -> removed single-model caveat.
- Per-layer ManifoldKV on NIAH-MK3: also collapses (0.65->0.03@b=0.5) -> closes the rebuttal.
- passage_count on Qwen2.5-14B: A_full 0.16 (above floor), gate closes 100% (mean D=-0.013), rho=0.01 -> confirmed 2nd non-RULER capacity-bound family (muted; MK3 stays the showcase). Limitation upgraded to result.
- THEORY (app:sharpened): tight scaling constant=1 + two-sided bound; necessary direction (cap-bound=>small D) under named margin assumption; A4 as named CUA assumption with O(H^2 eps) approx. Verified by me. Main-text limitation softened from "sketched" to "holds under named assumptions".
Paper: 9pp main, 40pp total, 0 undefined, 0 em-dashes.

Next: adversarial-reviewer panel (6 lenses) on final draft, then fix gaps.

## Tier 2 experiments integrated (2026-07-04, batches 3-4)
- Faithful ManifoldKV+AdaKV (manifoldkv_faithful_mk3.md): rescues single-key NIAH
  (0.88-0.94@2-4x, validates impl) but STILL collapses MK3 (0.65->0.23@50%, 0 below).
  MK3 capacity-bound even for competent geometry scorer -> strengthened scorer-relative claim.
- Matched-memory (matched_memory_comparison.md): crossover ~0.32-0.35 kept-KV; PAGE improves
  frontier above it, cannot compete below (3x max compression). HONEST: at matched ~30% memory,
  plain DBTrimKV BEATS wrapped PAGE; +23.3pp is a memory-spending win (14.5x cache). Integrated.
- FlashAttention cost (flashattn_deployment_cost.md): two-pass re-forward ~linear in T, drop
  O(H^2) indep of T; head-pairs 66/496/780/2016 (1.5B/Mistral/32B/70B). Confirms deployment limitation.
- Realistic workload (realistic_workload_partition.md): hotpotqa evict-robust, gate correctly
  opens (D=0.097, correct). Code tasks (lcc/repobench) small-N/incomplete. No realistic
  capacity-bound anchor found (still RULER-only, noted honestly).
Still pending: second_seed_robustness (running). Paper 9pp, clean, all honest integrations pushed.
