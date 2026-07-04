LAUNCHER=${LAUNCHER:-"python"}

# if LAUNCHER='slurm', then set the SLURM parameters
if [[ $LAUNCHER == "slurm" ]]; then
    LAUNCHER="sbatch scripts/wrapper_resub.sh python"
fi

#!/bin/bash

EVAL_DIR=$1
EVAL_MODEL=${EVAL_MODEL:-"qwen3-4b-instruct"}

# find all jsonl files in the eval dir
# and evaluate them one by one

find $EVAL_DIR -type f -name "*.jsonl" | while read -r file; do
    echo "Evaluating $file"
    echo "Using model $EVAL_MODEL"
    $LAUNCHER run_eval.py $EVAL_MODEL $file data/longmemeval_oracle.json
    echo "-----------------------------------"
done
