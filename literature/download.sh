#!/usr/bin/env bash
# Download all arXiv papers referenced in claude.md.
# Organized by thread. Filenames: <arxiv_id>_<short_slug>.pdf
set -u
cd "$(dirname "$0")"

UA="eff-nn-literature/1.0 (smishra@niser.ac.in)"
SLEEP=3  # arXiv robots policy: >=3s between requests

declare -a PAPERS=(
  # thread/dir | arxiv_id | slug
  "01_linear_attention_ssm|2605.11563|tcp_ssm_pole_based"
  "01_linear_attention_ssm|2602.04852|hidden_states_low_rank"
  "01_linear_attention_ssm|2506.04761|log_linear_attention"
  "01_linear_attention_ssm|2605.08301|hybrid_ssm_attention_priming"
  "01_linear_attention_ssm|2605.29157|local_linear_attention_ttr"
  "01_linear_attention_ssm|2605.08587|linear_attention_gates"
  "02_graph_ssm|2501.15461|mbagcn_oversmoothing"
  "02_graph_ssm|2402.08678|graph_mamba"
  "02_graph_ssm|2511.06756|dual_mamba_node_specific"
  "03_continuous_depth|2601.10007|continuous_depth_transformers"
  "03_continuous_depth|2605.21488|equilibrium_reasoners"
  "04_quantization|2602.02546|d2quant_sub4bit_ptq"
  "04_quantization|2605.12245|soar_nvfp4"
  "04_quantization|2512.04746|signround_v2"
  "05_kv_cache|2605.09649|learnable_global_kv_eviction"
  "05_kv_cache|2605.11478|fibquant_vq_random_access"
  "06_few_step_generation|2605.13724|anyflow_flow_map"
  "06_few_step_generation|2510.14974|pi_flow_imitation_distillation"
  "07_moe|2605.10933|deco_ondevice_moe"
  "07_moe|2605.12476|router_expert_geometric_coupling"
  "07_moe|2605.08575|intra_expert_neuron_sparsity"
  "07_moe|2510.05781|mixture_of_neuron_experts"
  "08_speculative_decoding|2605.14978|ppow_window_level_rl"
  "08_speculative_decoding|2605.10453|slimspec_low_rank_draft"
  "08_speculative_decoding|2605.08632|pard2_acceptance_length"
  "08_speculative_decoding|2510.05421|dvi_self_speculation"
  "09_reasoning_efficiency|2605.07315|later_latent_then_explicit"
  "09_reasoning_efficiency|2605.06285|latentrag"
  "09_reasoning_efficiency|2508.17196|budgetthinker_control_tokens"
  "10_edge_ondevice|2604.24785|single_board_benchmarking"
)

ok=0; fail=0; skip=0
for entry in "${PAPERS[@]}"; do
  IFS='|' read -r dir id slug <<< "$entry"
  out="${dir}/${id}_${slug}.pdf"
  if [[ -s "$out" ]]; then
    echo "[skip] $out"; skip=$((skip+1)); continue
  fi
  url="https://arxiv.org/pdf/${id}"
  echo "[get ] $id -> $out"
  if curl -sSL -A "$UA" --max-time 120 -o "$out.tmp" "$url" \
     && [[ -s "$out.tmp" ]] \
     && head -c 4 "$out.tmp" | grep -q '%PDF'; then
    mv "$out.tmp" "$out"; ok=$((ok+1))
  else
    echo "  FAILED: $url"
    rm -f "$out.tmp"; fail=$((fail+1))
  fi
  sleep "$SLEEP"
done
echo
echo "done: ok=$ok fail=$fail skip=$skip"
