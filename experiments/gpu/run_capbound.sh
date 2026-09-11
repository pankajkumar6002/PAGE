#!/usr/bin/env bash
# E9 (W1): is the capacity-bound class broader than one synthetic task?
#
# Tasks chosen by a-priori mechanism, not by result:
#   cwe, niah_multiquery  -- predicted capacity-bound
#   qa_2                  -- 29 real hard-negative distractors, predicted
#                            DILUTION-prone. The discriminating case: a
#                            distractor-count account predicts low D here, the
#                            near-tie-surface-form account predicts high D.
#   niah_single_1         -- no distractors, negative control
#
# Waits for E8 to finish first: only GPUs 0 and 2 are free (1 and 3 belong to
# other users) and contention already cost one 14B run to OOM.
set -uo pipefail

R2=/home/pankaj/Work/PAGE/page-kv
SCRIPTS="$R2/experiments/scripts"; OUT="$R2/experiments/results"; LOGDIR="$R2/experiments/logs"
export PAGE_SRC=/home/pankaj/Work/PAGE/page-kv/experiments/scripts
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate page-repro
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- gate 1: pre-registration intact -----------------------------------------
python - <<'PY' || exit 1
import hashlib, sys
p = "/home/pankaj/Work/PAGE/page-kv/preregistration/capacity_bound_second_family_prereg.md"
want = "7d336844cd6cd055d6eb27b5df42bf5ef2cb3ff07516349c586c289e49c0a59a"
got = hashlib.sha256(open(p, "rb").read()).hexdigest()
if got != want:
    sys.exit(f"E9 prereg hash mismatch\n  expected {want}\n  got      {got}")
print("E9 prereg verified:", got[:16], "...")
PY

# --- gate 2: wait for E8 to release the GPUs ---------------------------------
echo "waiting for E8 to finish before claiming GPUs 0,2 ..."
while pgrep -f "adakv_matrix.py --model Qwen/Qwen2.5-14B" >/dev/null 2>&1 \
   || pgrep -f "adakv_matrix.py --model Qwen/Qwen2.5-3B" >/dev/null 2>&1; do
  sleep 60
done
echo "E8 clear."

# --- gate 3: the runner must be validated on the released tasks first --------
# E9's tasks have no released reference, so the full-cache guard cannot fire on
# them. Correctness rests on the same runner having reproduced the released
# accuracies on the four known tasks.
python - <<'PY' || exit 1
import json, collections, sys, os
OUT = "/home/pankaj/Work/PAGE/page-kv/experiments/results"
REL = {"qwen15b":   {"niah_multikey_3":0.65,"vt":0.82,"fwe":0.22,"qa_1":0.74},
       "mistral7b": {"niah_multikey_3":0.99,"vt":1.00,"fwe":0.82,"qa_1":0.81},
       "qwen3b":    {"niah_multikey_3":0.93,"vt":1.00,"fwe":0.76,"qa_1":0.84},
       "qwen14b":   {"niah_multikey_3":1.00,"vt":1.00,"fwe":0.92,"qa_1":0.84}}
ok = 0
for slug, want in REL.items():
    f = os.path.join(OUT, f"adakv_4k_{slug}.jsonl")
    if not os.path.exists(f):
        print(f"  {slug}: no E8 run, skipped"); continue
    acc, seen = collections.defaultdict(list), set()
    for line in open(f):
        r = json.loads(line); k = (r["task"], r["id"])
        if k in seen: continue
        seen.add(k); acc[r["task"]].append(r["correct_full"])
    bad = [f"{t} {sum(v)/len(v):.3f}!={want[t]:.2f}"
           for t, v in acc.items() if t in want and abs(sum(v)/len(v)-want[t]) > 0.08]
    print(f"  {slug}: {'OK' if not bad else 'MISMATCH ' + ', '.join(bad)}")
    ok += not bad
if ok == 0:
    sys.exit("no E8 cell reproduced the released accuracies; do not run E9")
print(f"runner validated on {ok} released cell(s)")
PY

TASKS=cwe,niah_multiquery,qa_2,niah_single_1
BUDGETS=0.0625,0.125,0.25,0.375,0.5,0.625,0.75,0.875

echo "=== E9 capacity-bound probe ==="; date -u +"start: %FT%TZ"

# 14B first, alone across both free cards (it OOMs when it has to share).
echo "--- [1/4] qwen14b on GPUs 0,2 ---"
CUDA_VISIBLE_DEVICES=0,2 python "$SCRIPTS/adakv_matrix.py" \
  --model Qwen/Qwen2.5-14B-Instruct --slug capbound_qwen14b --config 4096 \
  --tasks "$TASKS" --max_examples 100 --budgets "$BUDGETS" \
  --tau 0.07 --floor_alpha 0.5 --device_map auto \
  --out "$OUT/capbound_4k_qwen14b.jsonl" > "$LOGDIR/capbound_qwen14b.log" 2>&1
echo "  exit=$?"

# The three smaller models fit one card each; run two at a time on 0 and 2.
echo "--- [2/4] qwen15b (GPU 0) + mistral7b (GPU 2) ---"
CUDA_VISIBLE_DEVICES=0 python "$SCRIPTS/adakv_matrix.py" \
  --model Qwen/Qwen2.5-1.5B-Instruct --slug capbound_qwen15b --config 4096 \
  --tasks "$TASKS" --max_examples 100 --budgets "$BUDGETS" \
  --tau 0.07 --floor_alpha 0.5 \
  --out "$OUT/capbound_4k_qwen15b.jsonl" > "$LOGDIR/capbound_qwen15b.log" 2>&1 &
p1=$!
CUDA_VISIBLE_DEVICES=2 python "$SCRIPTS/adakv_matrix.py" \
  --model mistralai/Mistral-7B-Instruct-v0.3 --slug capbound_mistral7b --config 4096 \
  --tasks "$TASKS" --max_examples 100 --budgets "$BUDGETS" \
  --tau 0.07 --floor_alpha 0.5 \
  --out "$OUT/capbound_4k_mistral7b.jsonl" > "$LOGDIR/capbound_mistral7b.log" 2>&1 &
p2=$!
wait $p1; echo "  qwen15b exit=$?"
wait $p2; echo "  mistral7b exit=$?"

echo "--- [4/4] qwen3b (GPU 0) ---"
CUDA_VISIBLE_DEVICES=0 python "$SCRIPTS/adakv_matrix.py" \
  --model Qwen/Qwen2.5-3B-Instruct --slug capbound_qwen3b --config 4096 \
  --tasks "$TASKS" --max_examples 100 --budgets "$BUDGETS" \
  --tau 0.07 --floor_alpha 0.5 \
  --out "$OUT/capbound_4k_qwen3b.jsonl" > "$LOGDIR/capbound_qwen3b.log" 2>&1
echo "  exit=$?"

date -u +"end: %FT%TZ"
for s in qwen15b qwen3b qwen14b mistral7b; do
  f="$OUT/capbound_4k_$s.jsonl"
  printf "  %-12s %s rows\n" "$s" "$([ -f "$f" ] && wc -l < "$f" || echo MISSING)"
done