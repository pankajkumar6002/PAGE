#!/bin/bash

LAUNCHER=${LAUNCHER:-"python"}
if [[ $LAUNCHER == "slurm" ]]; then
    LAUNCHER="sbatch scripts/wrapper.sh python"
elif [[ $LAUNCHER == "slurm_nmi" ]]; then
    LAUNCHER="sbatch scripts/wrapper_qos.sh python"
fi
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}

folder=$1
echo "Evaluating all .jsonl files in folder: $folder"
# find all files recursively in the folder that end with .jsonl
files=($(find $folder -name "*.json"))
echo "Found ${#files[@]} files to evaluate."

for file in "${files[@]}"; do
    echo "Evaluating file: $file"
    # check if file in the following patterns: <method>-<numbers>b-<numbers>l-<numbers>t.json
    if [[ $file =~ ([a-zA-Z0-9_]+)-([0-9]+)b-([0-9]+)l-([0-9]+)t\.json ]]; then
        method="${BASH_REMATCH[1]}"
        budget="${BASH_REMATCH[2]}"
        length="${BASH_REMATCH[3]}"
        max_new_tokens="${BASH_REMATCH[4]}"
        echo "Method: $method, Budget: $budget, Length: $length, max_new_tokens: $max_new_tokens"
        $LAUNCHER run_eval_mmdu.py --inference_backend=transformers --max_new_tokens 4096 --input_file $file $@
    else
        echo "File name does not match expected pattern. Skipping evaluation for this file."
    fi
done
