# Length-cut plan: ICLR 2026 build

**Status of this build** (`paper/iclr2026/main.pdf`, compiled 2026-07-02,
pdflatex + bibtex, zero errors):

- Total pages: **27**
- Main text (title through end of Conclusion, Section 9): ends **at the top of
  page 17** (~16.2 pages). Reproducibility + Ethics statements finish on p.17
  (excluded from the ICLR page limit), References run p.17–19, Appendices A–F
  p.19–27.
- **ICLR limit: 9 pages of main text at submission** → we must cut **~7.2
  pages** of main text. References and appendix are unlimited, so almost
  everything below is *moved*, not lost.

Page spans below are from this build's `.aux` (section start pages) and
`pdftotext`; floats are cited by their placed page.

| # | Section | Span (pages) | ~Length |
|---|---------|--------------|---------|
| — | Title + abstract | 1 | 0.9 |
| 1 | Introduction | 1–3 | 1.8 |
| 2 | Related work | 3–4 | 1.5 |
| 3 | The empirical partition | 4–6 | 2.2 (Tables 1–2 on p.5) |
| 4 | The gating method | 6–7 | 1.5 (Tables 3–5 p.6–7; Fig. 1 p.8) |
| 5 | Calibration recipe | 7–9 | 1.5 (Table 6 p.8; Fig. 2 p.9) |
| 6 | Theory | 9–11 | 2.2 |
| 7 | Experiments | 11–15 | 4.2 (Tables 7–12 p.11–15; Figs. 3–4 p.13–14) |
| 8 | Limitations | 15–16 | 1.4 |
| 9 | Conclusion | 16–17 | 0.8 |
| — | Repro + Ethics statements | 17 | 0.5 (does not count toward limit) |

## Non-negotiable keeps (per the paper's selling points)

Partition table (Table 1) · 4×4 matrix (Table 3) + cross-arch table (Table 4) ·
gate algorithm boxed equation + drop definition Eq. (D) · calibration recipe
(compressed) · scaling formula Eq. (scaling) + verification table (Table 10) ·
Mistral showcase (Fig. 4) · Limitations (compressed, not gutted) ·
Reproducibility statement (free: excluded from the limit).

## Section-by-section plan

### Title/abstract + Section 1 (Introduction), p.1–3 — **compress** — save ~1.0
- Abstract is ~0.7 page. Cut to ≤200 words: state partition, predictor,
  τ=0.07 transfer 7/8, grand-mean +22.9pp, Mistral 99→0 vs 89 showcase,
  scaling formula. Drop the per-model enumerations and the Llama caveat detail
  (one clause suffices). **~0.4 page.**
- Contributions list: each bullet currently restates numbers that reappear
  verbatim in Sections 3–7. Cut each to 2–3 lines, keep one headline number
  per bullet. Keep the "what we do NOT claim" paragraph (2 sentences — it is
  a reviewer-disarming asset). **~0.5 page.**
- Open-question (Q1/Q2) block: fold into one short paragraph. **~0.1 page.**

### Section 2 (Related work), p.3–4 — **compress + move detail to appendix** — save ~0.8
- Keep the five close neighbors but at 2–3 sentences each (claim + how we
  differ + pointer to the head-to-head section). DynamicKV paragraph is the
  longest and can be halved without losing the "orthogonal but overlapping"
  admission.
- "Concurrent 2026 diagnostics and wrappers" paragraph → 4–5 lines with
  citation clusters; move the per-paper distinctions to a new
  **Appendix: Extended related work**.
- "Older anchors" + "Adjacent" → merge into one 4-line paragraph.

### Section 3 (Partition), p.4–6 — **keep core, trim satellites** — save ~0.8
- **Keep**: Setup, Metric (Eq. rho + pre-registered 0.05 threshold),
  the partition paragraph, **Table 1 (partition) untouched**.
- Qwen-14B "re-entry" paragraph → 2 sentences (the scaling section already
  makes this point; keep the FWE 0.00→0.20 number).
- "Not a task-name heuristic" → 2 sentences.
- Distractor sweep: keep a 3-sentence summary (monotone D: 0.155→0.122→0.045,
  MK2/MK3 ranges non-overlapping) and **move Table 2 + the MK2 boundary-case
  paragraph to appendix** (per-task drop-ordering details are a designated
  appendix candidate). Random-eviction control pointer stays as is (already
  2 lines).

### Section 4 (Gating method), p.6–8 — **keep almost all; drop redundant float** — save ~0.8
- **Keep verbatim**: boxed GatedEvict conditional, head-agreement-drop
  definition Eq. (D), **Table 3 (4×4 matrix)**, **Table 4 (cross-arch)**, the
  Mistral showcase paragraph.
- **Drop Figure 1 (p.8)**: the bar chart is a strict visual duplicate of
  Table 3; move to appendix if sentimental. **~0.45 page.**
- Table 5 (best-plain vs best-gated) → one sentence ("best gated beats best
  plain by +13.2 to +28.5pp on every model") + table to appendix. **~0.25.**
- "Why the deltas vary" → 3 sentences; the PyramidKV-variant and 14B-H2O
  two-pass caveats move to the reproduction-notes appendix (App. F). **~0.1.**

### Section 5 (Calibration recipe), p.7–9 — **compress (must survive)** — save ~0.8
- Recipe enumerate → 3 compact lines + Eq. (tau); keep "40 prefills, no
  labels, <5 min".
- Keep **Table 6** (it is the recipe's only evidence and is small).
- τ-sensitivity: keep ONE sentence ("plateau τ∈[0.055,0.10], Δ varies <1pp,
  FP+FN ≤9%; App. X") and **move Figure 2 (p.9) + the long sweep paragraph to
  appendix** (designated candidate). **~0.6 page.**
- "Status" paragraph → 1 sentence appended to the validation paragraph.

### Section 6 (Theory), p.9–11 — **keep statements, move derivations** — save ~1.0
- 6.1 Mechanism: compress to one paragraph — cite Bui Prop. 3.1/Cor. 3.2,
  define δ_t in-line (Eq. dilution stays, it is referenced later). The
  full setup notation moves next to App. D which already restates it.
- 6.2 Partition-follows-from-mechanism: keep the two-bullet D/C definition
  (it is short and load-bearing).
- 6.3 Context-length scaling: **keep Eq. (scaling), the informal Proposition,
  and the 3-effect list (compressed to one line each)**; move the SNR algebra
  (pre/post-eviction SNR display equations and the ρ_R/γ monotonicity
  argument) into App. D, which already contains the same derivation — the
  main-text copy is near-duplicate. Delete the proof sketch paragraph
  (App. D pointer suffices). **~0.7.**
- 6.4 Sufficient condition: keep Eq. (bridge) + one honesty sentence
  ("constants not tight, converse unproven, App. E") — cut the two long
  framing paragraphs. **~0.3.**

### Section 7 (Experiments), p.11–15 — **largest cut** — save ~2.9
- 7.1 Setup: 3 lines (mostly repeats Section 3 setup).
- 7.2 **Table 7 (Qwen-3B budget curve, p.11) → appendix** (designated
  candidate); keep one sentence ("Δ grows monotonically from +0.08 to +0.42
  as b shrinks; App. X"). **~0.45.**
- 7.3 8-cell scaling table (Table 8, p.12): **keep** — it is the only
  cross-(model,context) gating evidence and is compact.
- 7.4 Per-task drop ordering: **Table 9 (p.12) → appendix**; keep 3-sentence
  summary (NIAH-MK3 smallest D in every Qwen/Mistral cell; 2nd-smallest on
  Llama family). The long Llama-transfer paragraph (~0.5 page) is 70%
  redundant with Limitations — keep 4 sentences here (Yi gain is fallback-
  driven, not partition transfer), move the per-task dissection to appendix.
  **~0.8.**
- 7.5 Scaling verification: **keep Table 10 + intro prose (must-keep)**;
  **Figure 3 (p.13) → appendix** — it re-plots Table 10. **~0.4.**
- 7.6 Mistral showcase: **keep Figure 4 + prose untouched (must-keep).**
- 7.7–7.9 Per-layer ablation, random control, gate cost: merge into one
  "Ablations and cost" paragraph, 5 lines total, with App. A/B pointers
  (7.8 already just points to App. A). Keep the 50ms/<1% cost number.
  **~0.3.**
- 7.10 CapKV head-to-head → **2-sentence summary** (+1.5pp qasper / +8.3pp
  triviaqa at b=0.5, gain concentrated on gate-closed subset; directional,
  small N) — **Table 11 + re-implementation details + hedging paragraph to
  appendix**. **~0.5.**
- 7.11 DBTrimKV head-to-head → **2-sentence summary** (+0.233 at matched
  budget on Qwen3-4B RULER 4K; caveat: gate fires 1.7%, win dominated by
  full-KV fallback, quality-not-memory win) — **Table 12 + caveat paragraph
  to appendix**. The caveat clause must survive in the main text — it is the
  honesty selling point. **~0.5.**

### Section 8 (Limitations), p.15–16 — **compress, do not gut** — save ~0.8
Keep every limitation as a named item; cut prose per item:
- Keep at 2–3 sentences each: sketched theorem; Llama-family τ non-transfer
  (keep the "drop-vs-ρ prediction is inverted on Yi" sentence — the honesty
  is a selling point); scoped-to-attention-score evictors (ManifoldKV);
  LongBench Δ≈0 neutrality incl. the −2.2pp worst cell (the only negative
  number — keep it).
- Fold into single sentences: protection-ablation mismatch; single-seed/no-CI;
  Qwen-3B-16K needs recipe (already stated in Sec. 5); DBTrimKV-on-Qwen3,
  4-not-6 evictors, RULER-only → one combined "scope" item.
Result: ~0.6 page instead of 1.4.

### Section 9 (Conclusion), p.16–17 — **compress** — save ~0.4
One paragraph (partition + 50ms statistic + wrapper positive everywhere +
Mistral number) + one 2-sentence open-questions note. The "so the next paper
does not re-derive them" framing can survive in half the words.

### Repro + Ethics statements — **keep as-is** (excluded from the 9-page limit
under ICLR rules; they sit after the conclusion, before references).

## Bookkeeping

| Section | Save (pages) |
|---|---|
| Abstract + Intro | 1.0 |
| Related work | 0.8 |
| Partition | 0.8 |
| Method | 0.8 |
| Recipe | 0.8 |
| Theory | 1.0 |
| Experiments | 2.9 |
| Limitations | 0.8 |
| Conclusion | 0.4 |
| **Total** | **~9.3** |

Needed: ~7.2. The ~2-page surplus is deliberate: float reflow eats savings
(each moved table/figure frees slightly less than its printed height), and it
leaves room to keep Table 6 and Table 8 if the first compile after cutting
lands at 8.5 pages. Apply cuts in this order (cheapest signal loss first):

1. Move floats: Fig. 1, Fig. 2, Fig. 3, Tables 2, 5, 7, 9, 11, 12 → appendix
   (mechanical, ~3.5 pages).
2. Compress abstract, contributions, related work (~1.8 pages).
3. Theory derivation → App. D dedup; 6.4 framing cut (~1.0 page).
4. Limitations + conclusion compression (~1.2 pages).
5. Only if still >9: shorten Section 3 satellites and recipe prose (~0.8).

New appendix sections to create: Extended related work; Distractor sweep &
MK2 boundary case; Best-plain-vs-best-gated + Qwen-3B budget curve; Per-task
drop ordering & Llama transfer detail; τ-sensitivity; CapKV & DBTrimKV
head-to-head tables. All are additive — appendix is unlimited.
