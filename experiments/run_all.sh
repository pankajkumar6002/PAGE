#!/usr/bin/env bash
# Zero-GPU reanalyses for the paper's revision round.
#
# Reads the per-input logs from experiments/results (override with
# PAGE_RESULTS) and writes tables to experiments/results
# (override with PAGE_OUT). Every script exits non-zero if its numbers stop
# matching the ones promoted into the paper.
#
#   ./run_all.sh            run everything, tee to logs/
#
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$HERE/scripts"
LOGDIR="$HERE/logs"
mkdir -p "$LOGDIR" "$HERE/results"

PY="${PY:-python3}"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUNLOG="$LOGDIR/run_all-$STAMP.log"
ln -sfn "$RUNLOG" "$LOGDIR/latest.log"

STEPS=(
  audit_budget1_identity   # P0-1: r(x) >= 0 is an identity
  harm_rate                # P0-1: the replacement metric
  per_task_delta_matrix    # P0-2: Delta is one measurement replicated 16x
  mixture_sweep            # P1-6: Delta vs capacity-bound workload share
  layer_subsample          # P1-7: does the gate signal need every layer?
  layer_profile            # P-1:  the a_l profile A3 asserts
  fig1_panel3              # Fig 1 panel 3, drawn from the measured curves
  task_label_oracle        # DA-6: does a task-label oracle match PAGE?
  seed_variance            # P3-1: is the headline one lucky input draw?
  heldout_ablation         # P3-2: does D still win off the fitting cell?
  head_pair_subsample      # P3-3: how much of the H^2 term is load-bearing?
)

# The three steps above re-analyse this round's GPU runs, which paths.py reads
# from PAGE_DATA (default experiments/results) rather than from the released
# logs. measure_kappa is not listed: it needs a GPU and a model download, so it
# is not a re-analysis and cannot run here.

fail=0
{
  echo "run_all $STAMP"
  echo "PAGE_RESULTS=${PAGE_RESULTS:-<default>}"
  echo "PAGE_OUT=${PAGE_OUT:-<default>}"
  echo
  for s in "${STEPS[@]}"; do
    echo "=============================================================="
    echo "== $s"
    echo "=============================================================="
    if (cd "$SCRIPTS" && "$PY" "$s.py"); then
      echo "-- $s: PASS"
    else
      echo "-- $s: FAIL (exit $?)"
      fail=1
    fi
    echo
  done
  echo "=============================================================="
  if [ "$fail" -eq 0 ]; then
    echo "ALL CHECKS PASSED"
  else
    echo "SOME CHECKS FAILED - do not promote numbers into the paper"
  fi
} 2>&1 | tee "$RUNLOG"

exit "$fail"