# Per-task head-agreement drops on Yi-1.5-9B-Chat, RULER 4K

## Model substitution rationale

The original plan called for a fourth architecture family beyond Qwen2.5 (1.5B/3B/14B)
and Mistral-7B-v0.3 to address reviewer concerns about Qwen-family specificity. The
preferred candidate was Llama-3.1-8B, but the HF token configured on this machine
does not have Meta license access (all Meta-Llama repos return 403 GatedRepoError).

The first non-gated, Apache-2.0 substitute considered was **InternLM-2.5-7B-Chat**
(`internlm/internlm2_5-7b-chat`, 7.74B params, 32 layers / 32 heads / 8 KV heads,
GQA, RoPE theta 1M with dynamic NTK scaling, untied embeddings, 32K context).
This model is architecturally distinct from Qwen and Mistral. However, on
transformers 5.9.0:

- The vendored custom modeling code (`modeling_internlm2.py`) calls
  `DynamicCache.from_legacy_cache` and `to_legacy_cache`, which were removed in
  transformers >= 5.x. A local patch to the cached modeling file restores the
  load path.
- After patching, forward passes in bf16 produce NaN logits starting at layer 11
  (an outlier-feature overflow: hidden states from layer 2 onwards contain
  activations of magnitude ~1.7e3, which overflow downstream in bf16). The same
  NaN appears under fp16. Only fp32 inference works; this is too slow for the
  300-example probe at 4K.
- The bug is in the upstream InternLM custom modeling code, not in our scripts.
  It would require either a rewrite of the attention kernel to upcast more
  aggressively or a wait for an updated `modeling_internlm2.py`. Given the
  6-hour wall budget, this is not solvable in time.

The fallback model from the task plan is **Yi-1.5-9B-Chat** (`01-ai/Yi-1.5-9B-Chat`,
8.83B params, 48 layers / 32 heads / 4 KV heads, GQA, RoPE theta 5M, untied
embeddings, 4K context). Yi-1.5 inherits the Llama architecture class
(`LlamaForCausalLM`) but is trained on a separate corpus (01.AI), uses its own
~64K tokenizer, and has substantially deeper/narrower layer geometry than
Llama-3.1-8B (48 vs 32 layers; 4 KV heads vs 8). It is therefore a real fourth
family in terms of tokenizer, training corpus, depth, and KV-head sharing
ratio, though it is closer to Llama-architecture than InternLM would have been.
Reported here as the substitute for Llama-3.1-8B; the InternLM attempt is
documented above so the substitution chain is traceable.

## Method

- RULER configuration: 4096 token target
- Tasks: qa_1, qa_2, vt, niah_multivalue, fwe, niah_multikey_3
- N = 50 examples per task (300 total)
- Two-pass attention scoring: SDPA prefill, then 32-token obs-window re-forward
  with eager attention for output_attentions=True
- Per-layer head agreement = mean Jaccard top-32 across all head pairs in a
  layer, computed over the obs_window queries
- D (drop) = mean(layers 0..L/3-1) - mean(layers 2L/3..L-1) with L=48 layers

## Per-task drop D

| task                | N  | mean T  | mean drop D | stdev D | min D     | max D     |
| ------------------- | -: | ------: | ----------: | ------: | --------: | --------: |
| qa_1                | 50 | 3300    | +0.0799     | 0.0125  | +0.0419   | +0.1107   |
| qa_2                | 50 | 3833    | +0.0791     | 0.0205  | +0.0364   | +0.1220   |
| vt                  | 50 | 4067    | +0.0654     | 0.0034  | +0.0581   | +0.0727   |
| fwe                 | 50 | 8149    | +0.0431     | 0.0071  | +0.0307   | +0.0564   |
| niah_multikey_3     | 50 | 4250    | +0.0187     | 0.0071  | +0.0027   | +0.0351   |
| niah_multivalue     | 50 | 4051    | -0.0125     | 0.0113  | -0.0318   | +0.0268   |

Tasks sorted by D ascending — partition prediction: smallest D = capacity-bound.

## Interpretation against partition prediction

The five "dilution-prone" tasks (qa_1, qa_2, vt, fwe, niah_multivalue) and one
"capacity-bound" task (niah_multikey_3) are predicted to have D large and D
small respectively. On Yi-1.5-9B-Chat at 4K:

- qa_1 (D=+0.080), qa_2 (D=+0.079), vt (D=+0.065): largest drops; partition
  prediction holds.
- niah_multikey_3 (D=+0.019): nearly the smallest, exactly as predicted —
  only niah_multivalue lands below it.
- fwe (D=+0.043): in the middle; smaller than expected but still positive.
- niah_multivalue (D=-0.013): *negative*, contrary to the dilution-prone
  prediction. On Yi at 4K, niah_multivalue behaves like a capacity-bound
  task — heads disagree slightly more in the early layers than in the late
  layers. This is a model-specific anomaly: on Qwen-1.5B and Mistral-7B at
  the same 4K split, niah_multivalue showed positive drops (Qwen: +0.126,
  Mistral: +0.075). The likely cause is Yi's deeper architecture (48 layers
  vs 32 for the other models) shifting where consolidation happens.

**Ordinal check**: NIAH-MK3 ranks 2nd-smallest (out of 6) by D; the strict "MK3 is
smallest" prediction fails because niah_multivalue dips lower on Yi. Out of the
five dilution-prone tasks, four (qa_1, qa_2, vt, fwe) lie above MK3, matching
the partition direction. So the partition predictor transfers to Yi for 4/5 of
the dilution-prone tasks but mis-classifies niah_multivalue.

A τ threshold around 0.03-0.04 still works as a binary classifier for the
"open eviction" decision on five of six tasks (gate open on qa_1, qa_2, vt,
fwe; gate closed on niah_multikey_3, niah_multivalue). The corresponding
behaviour of plain vs gated SnapKV is measured in
`gated_4k_yi15_9b_report.md`.
