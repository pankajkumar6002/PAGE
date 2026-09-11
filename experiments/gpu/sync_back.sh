#!/usr/bin/env bash
# Pull GPU results and logs back from the run host into this checkout, so the
# jsonl produced on the remote GPU host sits beside the zero-GPU outputs that
# run_all.sh already asserts against.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL="$(cd "$HERE/../.." && pwd)"

: "${HOST:?set HOST=user@remote-gpu-host}"
: "${REMOTE:?set REMOTE=path/to/page-kv on the remote host}"

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
