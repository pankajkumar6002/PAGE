# Literature index

Papers referenced in `../claude.md` (May 2026 anchors). One-line descriptions
copied verbatim from CLAUDE.md threads. Files are named
`<arxiv_id>_<short_slug>.pdf` inside each thread directory.

## 01 Linear attention / SSM design
- **2605.11563** TCP-SSM — pole-based, control-theoretic view of SSM scan.
- **2602.04852** Hidden states low-rank after training; ~50% of key/query channels prunable.
- **2506.04761** Log-linear attention.
- **2605.08301** Hybrid SSM-attention priming.
- **2605.29157** Local Linear Attention via test-time regression.
- **2605.08587** Linear-attention gates tuned empirically rather than derived from the regression objective.

## 02 Graph + state space
- **2501.15461** MbaGCN for over-smoothing.
- **2402.08678** Graph Mamba.
- **2511.06756** Dual Mamba node-specific.

## 03 Continuous depth / Neural ODE
- **2601.10007** Continuous-depth transformers with learned control.
- **2605.21488** Equilibrium reasoners linking test-time scaling to attractor convergence.

## 04 Quantization
- **2602.02546** D2Quant — sub-4-bit weight-only PTQ.
- **2605.12245** SOAR for NVFP4.
- **2512.04746** SignRoundV2.

## 05 KV cache
- **2605.09649** Learnable global eviction that improves reasoning.
- **2605.11478** FibQuant vector quantization for random-access cache.

## 06 Few-step generation
- **2605.13724** AnyFlow — flow-map distillation for any-step video.
- **2510.14974** pi-Flow — imitation distillation.

## 07 MoE
- **2605.10933** DECO — dense-comparable on-device MoE.
- **2605.12476** Router-expert geometric coupling; load-balancing losses that damage router-expert structure.
- **2605.08575** Intra-expert neuron sparsity (<2% active routed-expert neurons retain capacity with a shared expert).
- **2510.05781** Mixture of Neuron Experts.

## 08 Speculative decoding
- **2605.14978** PPOW — window-level RL drafting (acceptance length objective).
- **2605.10453** SlimSpec — low-rank draft head.
- **2605.08632** PARD-2 — acceptance-length objective.
- **2510.05421** DVI — training-aware self-speculation.

## 09 Reasoning efficiency
- **2605.07315** LaTER — latent-then-explicit reasoning.
- **2605.06285** LatentRAG — ~90% latency cut.
- **2508.17196** BudgetThinker — control tokens.

## 10 Edge / on-device
- **2604.24785** Single-board benchmarking.
- *On-Device LLMs State of the Union 2026* — web report at `v-chandra.github.io/on-device-llms` (not on arXiv; not downloaded).

## 11 Concurrent neighbors (added 2026-06-01 from arxiv re-search)

Papers found while checking for concurrent unified-eviction-theory work. Important for `problem_statement.md` v0.2.

- **2603.18446** UT-ACA — uncertainty-triggered adaptive context allocation. Closest H2 competitor on the KV axis. Must read in week 1.
- **2602.08585** LU-KV — task-agnostic per-head budget allocation via convex-hull relaxation.
- **2602.03203** ForesightKV — trained predictive eviction for reasoning models.
- **2602.10238** Learning to Evict — RL-trained eviction with per-head policies.
- **2604.14853** Adaptive Test-Time Compute via Constrained Policy Optimization — reasoning-axis budget allocation.
- **2603.12634** Budget-Aware Value Tree Search — adjacent (search-based reasoning).
- **2602.13804** Vashista Sparse Attention — theoretical sibling (exponential mass on irrelevant tokens).
- **2506.16640** Long-Context Generalization with Sparse Attention — ICLR 2026, attention-dispersion framing.
- **2506.08371** Positional Contrastive Decoding — posterior salience attenuation.
- **2511.05313** Attention and Compression for Controllably Efficient LMs — quality-compute tradeoff framing.

## 12 ICLR style exemplars (added 2026-07-04)
Full PDFs + extracted text in `12_iclr_style_exemplars/`. Accepted ICLR papers
in the KV/attention-efficiency space, chosen to match our diagnostic-then-method
structure. Studied for tone/style, not cited unless already in main.bib.
- **2309.17453** StreamingLLM (ICLR 2024) — attention-sink observation -> method; abstract leads with phenomenon then named framework + speedup.
- **2310.01801** FastGen "Model Tells You What to Discard" (ICLR 2024) — closest structural twin: profile attention structure -> adaptive plug-and-play compression, no retraining ("diagnose-before-compress").
- **2410.10819** DuoAttention (ICLR 2025) — retrieval vs streaming heads; head-type partition -> per-head cache policy; leads with memory/latency numbers.
