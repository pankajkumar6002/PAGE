# B8 raw data sources (downloaded 2026-11-10)

- nq30_gold_at_{0,4,9,14,19,24,29}.jsonl.gz, qa.prompt, metrics.py, liu_prompting.py:
  https://github.com/nelson-liu/lost-in-the-middle, branch main, paths qa_data/30_total_documents/,
  src/lost_in_the_middle/prompts/qa.prompt, src/lost_in_the_middle/{metrics,prompting}.py (Liu et al., TACL 2024).
- babilong_{qa2,qa3}_{4k,8k}.json: https://huggingface.co/datasets/RMT-team/babilong, branch main, data/<qa>/<len>.json.
  babilong_prompts.py, babilong_metrics.py: https://github.com/booydar/babilong, branch main, babilong/ (Kuratov et al., NeurIPS 2024).
- Downloaded as data. No loading script or downloaded code is executed. The runner holds verbatim copies of the
  prompt templates and scoring functions, checked equal to these files (DEFAULT_PROMPTS compared via ast.literal_eval).
