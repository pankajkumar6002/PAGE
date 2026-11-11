#!/usr/bin/env bash
# Download the third-party data used by the revision's Table 45 and 48 runs (b8_neartie.py and b7_leval.py)
# from pinned public sources, then check every file against the SHA256SUMS files in this folder.
#
#   b8_raw/  NQ-open with 30 retrieved passages, prompt and scoring code from "Lost in the Middle"
#            (Liu et al., TACL 2024; github.com/nelson-liu/lost-in-the-middle, MIT licence), and
#            BABILong qa2/qa3 at 4k and 8k with its prompt and scoring code (Kuratov et al., NeurIPS 2024;
#            github.com/booydar/babilong, Apache-2.0; data from huggingface.co/datasets/RMT-team/babilong).
#   leval/   the six L-Eval exam tasks used here (An et al., ACL 2024; huggingface.co/datasets/L4NLP/LEval,
#            GPL-3.0), at the commit recorded in leval/COMMIT.
#
# Nothing downloaded here is executed. The two third-party .py files of each project are fetched only so
# that b8_raw/SHA256SUMS, which the B8 pre-registration hashes, can be checked in full; the runners hold
# their own verbatim copies of the prompt templates and scoring functions (see THIRD_PARTY_NOTICES.md).
#
# Usage: bash experiments/data/revision/fetch.sh     (about 170 MB; needs curl and sha256sum or shasum)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LITM=29b8a6d042ce29abccee3db1a73171a107d7e6af        # github.com/nelson-liu/lost-in-the-middle
BABI_CODE=7a6efee29f5cac03c3c410e6799c80fd2ffe3610   # github.com/booydar/babilong
BABI_DATA=ee0d588794c7ac098062ee0d247c733d62e94fe2   # huggingface.co/datasets/RMT-team/babilong
LEVAL="$(tr -d '[:space:]' < "$HERE/leval/COMMIT")"  # huggingface.co/datasets/L4NLP/LEval

get() {  # url, output path; skips files that are already complete
  local url="$1" out="$2"
  if [ -s "$out" ]; then return 0; fi
  mkdir -p "$(dirname "$out")"
  curl -fL --retry 3 -o "$out.part" "$url"
  mv "$out.part" "$out"
}

check() {  # directory holding a SHA256SUMS file, then optional extra flags (--ignore-missing)
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
# leval/SHA256SUMS is L-Eval's full file list as recorded before the runs (the B7 pre-registration hashes it).
# Only the six exam tasks above are used and fetched, so the others are skipped here.
check "$HERE/leval" --ignore-missing
echo "All files present and verified."
