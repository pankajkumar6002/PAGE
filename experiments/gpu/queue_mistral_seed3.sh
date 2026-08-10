#!/usr/bin/env bash
# Wait for the three Qwen-14B seed runs to finish, then run Mistral seed 3 on
# the freed card.
#
# Why it waits rather than running now: Mistral-7B at 4K with full eager
# attention peaks above 48 GiB, so it needs an 80 GiB card. Seeds 1 and 2 were
# measured here with these exact settings, and switching to --two_pass or to
# the 48 GiB host to make it fit would make seed 3 incomparable with them,
# which defeats the point of measuring seed variance.
set -uo pipefail
R="$HOME/Work/PAGE/page-kv"
RES="$R/experiments/results"

echo "waiting for qwen14b seeds 1-3 (3600 rows each)..."
while :; do
  done_n=0
  for s in 1 2 3; do
    n=$(wc -l < "$RES/rebase_4k_qwen14b_seed${s}.jsonl" 2>/dev/null || echo 0)
    [ "$n" -ge 3600 ] && done_n=$((done_n+1))
  done
  [ "$done_n" -ge 3 ] && break
  sleep 120
done
echo "qwen seeds complete; starting mistral seed 3 on gpu 0"
GPU=0 "$R/experiments/gpu/run_mistral_seed3.sh"
