#!/usr/bin/env bash
# E11 (W1 / Q1 / DA-2): prevalence gap subtasks + AgentLongBench agentic slice.
#
# The 9 ingestible LongBench subtasks need NO GPU (prevalence_survey.py reads
# released logs). This script fills the gap subtasks that have no released log:
#   - LongBench multi-hop / summarization (2wikimqa, musique, gov_report)
#   - AgentLongBench agentic tool-call traces (out-of-fitting-range, 32k bucket)
#
# NOT auto-queued. Launch only when the user says a GPU is free (round rule in
# PENDING.md: "Only GPUs 0 and 2 are ours").
set -uo pipefail

R2=/home/pankaj/Work/PAGE/new-exp-page-kv-r2
SCRIPTS="$R2/experiments/scripts"
OUT="$R2/experiments/results"
LOGDIR="$R2/logs"
export PAGE_SRC=/home/pankaj/Work/PAGE/page-kv/experiments/scripts
GPU="${1:-0}"                     # pass a free card index as $1

mkdir -p "$OUT" "$LOGDIR"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate page-repro

# Gate: refuse to run if the E11 pre-registration was edited after hashing.
python - <<'PY' || exit 1
import hashlib, sys
p = "/home/pankaj/Work/PAGE/new-exp-page-kv-r2/preregistration/prevalence_survey_prereg.md"
raw = open(p, "rb").read()
i = raw.find(b"## Addendum")
frozen = raw[:i-6] if i != -1 else raw
got = hashlib.sha256(frozen).hexdigest()
want = open(p.replace(".md", ".sha256")).read().split()[0]
if got != want:
    sys.exit(f"E11 prereg hash mismatch\n  want {want}\n  got  {got}")
print("[gate] E11 prereg intact:", got[:16])
PY

MODEL="Qwen/Qwen2.5-14B-Instruct"    # match the ingest panel's model (single-model pool)
SLUG="qwen14b"
# Realistic-workload budget grid (5 budgets), matching longbench_realistic_* logs.
BUDGETS="1.0,0.5,0.25,0.125,0.0625"

# --- (1) LongBench gap subtasks (configs verified present in Xnhyacinth/LongBench) ---
for TASK in 2wikimqa musique gov_report; do
  echo "[e11] launching LongBench/$TASK on GPU $GPU"
  python "$PAGE_SRC/longbench_gating.py" \
    --model "$MODEL" --tasks "$TASK" \
    --budgets "$BUDGETS" --max_examples 100 --tau 0.07 \
    --gpu "$GPU" --attn_impl sdpa \
    --out "$OUT/longbench_${TASK}_${SLUG}.jsonl" \
    > "$LOGDIR/e11_${TASK}_${SLUG}.log" 2>&1
done

# --- (2) AgentLongBench agentic slice (32k bucket; OUT OF FITTING RANGE) ---
# longbench_gating skips inputs above --max_context_tokens; 32k traces are ~2x
# the 16K max, so we RAISE the cap to admit them and mark the result
# out-of-range. Loader is injected by monkeypatching load_dataset, mirroring
# longbench_capkv_seed2.py.
echo "[e11] launching AgentLongBench (32k, out-of-range) on GPU $GPU"
PAGE_SRC="$PAGE_SRC" ALB_SETTING=ki-c ALB_CTXLEN=32k \
python - "$MODEL" "$SLUG" "$GPU" "$OUT" <<'PY' \
  > "$LOGDIR/e11_agentlongbench_${SLUG}.log" 2>&1
import os, sys
sys.path.insert(0, os.environ["PAGE_SRC"])
sys.path.insert(0, "/home/pankaj/Work/PAGE/new-exp-page-kv-r2/experiments/scripts")
model, slug, gpu, out = sys.argv[1:5]
import agentlongbench_loader as alb
import longbench_gating as lg

rows = alb.load_agentlongbench(setting=os.environ["ALB_SETTING"],
                               ctxlen=os.environ["ALB_CTXLEN"],
                               max_examples=50)
# Inject as the dataset longbench_gating loads. It calls
# load_dataset("Xnhyacinth/LongBench", task, split="test").select(...); we
# replace that with our pre-materialized rows keyed by _task.
_tasks = sorted({r["_task"] for r in rows})
class _DS(list):
    def select(self, idxs): return _DS([self[i] for i in idxs])
def fake_load_dataset(name, task=None, split=None, **kw):
    return _DS([r for r in rows if r["_task"] == task])
lg.load_dataset = fake_load_dataset

sys.argv = ["longbench_gating.py",
            "--model", model, "--tasks", ",".join(_tasks),
            "--budgets", "1.0,0.5,0.25,0.125,0.0625",
            "--max_examples", "50", "--tau", "0.07", "--gpu", gpu,
            "--attn_impl", "sdpa",
            "--max_context_tokens", "40000",   # admit 32k traces (OUT OF RANGE)
            "--out", os.path.join(out, f"agentlongbench_32k_{slug}.jsonl")]
lg.main()
print("[e11] AgentLongBench done (OUT OF FITTING RANGE; gate-closure only).")
PY

echo "[e11] all gap cells launched. Re-run prevalence_survey.py after they finish"
echo "      to fold the LongBench gaps into the pooled f; AgentLongBench is"
echo "      analyzed separately as out-of-range (do NOT pool it)."