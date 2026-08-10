#!/usr/bin/env bash
# Pull GPU results and logs back from the run host into this checkout, so the
# jsonl produced on sateri sits beside the zero-GPU outputs that run_all.sh
# already asserts against.
set -uo pipefail

HOST=${HOST:-pankaj@10.10.0.190}
LOCAL=/home/pankaj/Work/PAGE/page-kv
REMOTE=Work/PAGE/page-kv

mkdir -p "$LOCAL/experiments/results" "$LOCAL/logs"

echo ">> results"
# --min-size=1 stops an empty file from a failed remote run clobbering a
# good local result, which is how the Qwen-14B ablation was lost once.
rsync -az --info=stats1 --min-size=1 \
  "$HOST:$REMOTE/experiments/results/" "$LOCAL/experiments/results/" | tail -2

echo ">> logs"
rsync -az --info=stats1 \
  "$HOST:$REMOTE/logs/" "$LOCAL/logs/" | tail -2

echo ">> new/updated jsonl:"
find "$LOCAL/experiments/results" -name "*.jsonl" -mmin -600 -printf "   %f  %s bytes\n" | sort
