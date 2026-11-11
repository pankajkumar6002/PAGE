#!/usr/bin/env bash
# Downloads and verifies the third-party data for Tables 45 and 48. Sources and licences: THIRD_PARTY_NOTICES.md.
# Usage: bash experiments/data/revision/fetch.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LITM=29b8a6d042ce29abccee3db1a73171a107d7e6af
BABI_CODE=7a6efee29f5cac03c3c410e6799c80fd2ffe3610
BABI_DATA=ee0d588794c7ac098062ee0d247c733d62e94fe2
LEVAL="$(tr -d '[:space:]' < "$HERE/leval/COMMIT")"

get() {
  local url="$1" out="$2"
  if [ -s "$out" ]; then return 0; fi
  mkdir -p "$(dirname "$out")"
  curl -fL --retry 3 -o "$out.part" "$url"
  mv "$out.part" "$out"
}

check() {
  local dir="$1"; shift
  if command -v sha256sum >/dev/null 2>&1; then
    (cd "$dir" && sha256sum -c "$@" SHA256SUMS)
  else
    (cd "$dir" && shasum -a 256 -c "$@" SHA256SUMS)
  fi
}

LITM_RAW="https://raw.githubusercontent.com/nelson-liu/lost-in-the-middle/$LITM"
for p in 0 4 9 14 19 24 29; do
  get "$LITM_RAW/qa_data/30_total_documents/nq-open-30_total_documents_gold_at_$p.jsonl.gz" "$HERE/b8_raw/nq30_gold_at_$p.jsonl.gz"
done
get "$LITM_RAW/src/lost_in_the_middle/prompts/qa.prompt" "$HERE/b8_raw/qa.prompt"
get "$LITM_RAW/src/lost_in_the_middle/metrics.py" "$HERE/b8_raw/metrics.py"
get "$LITM_RAW/src/lost_in_the_middle/prompting.py" "$HERE/b8_raw/liu_prompting.py"

BABI_RAW="https://raw.githubusercontent.com/booydar/babilong/$BABI_CODE"
get "$BABI_RAW/babilong/prompts.py" "$HERE/b8_raw/babilong_prompts.py"
get "$BABI_RAW/babilong/metrics.py" "$HERE/b8_raw/babilong_metrics.py"
for qa in qa2 qa3; do
  for len in 4k 8k; do
    get "https://huggingface.co/datasets/RMT-team/babilong/resolve/$BABI_DATA/data/$qa/$len.json" "$HERE/b8_raw/babilong_${qa}_$len.json"
  done
done

for task in tpo quality coursera topic_retrieval_longchat gsm100 sci_fi; do
  get "https://huggingface.co/datasets/L4NLP/LEval/resolve/$LEVAL/LEval/Exam/$task.jsonl" "$HERE/leval/Exam/$task.jsonl"
done

check "$HERE/b8_raw"
check "$HERE/leval" --ignore-missing
echo "All files present and verified."
