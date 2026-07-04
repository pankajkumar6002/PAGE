#!/bin/bash

# export $(cat .env | xargs)

LAUNCHER=${LAUNCHER:-"python"}
if [[ $LAUNCHER == "slurm" ]]; then
    LAUNCHER="sbatch scripts/wrapper.sh python"
elif [[ $LAUNCHER == "slurm_nmi" ]]; then
    LAUNCHER="sbatch scripts/wrapper_qos.sh python"
fi
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}

max_new_tokens=${MAX_NEW_TOKENS:-4096}
budgets=(128)
model_path=${MODEL_PATH:-"Qwen/Qwen3-VL-4B-Instruct"}

basemodel_name=$(basename $model_path)

echo "Using model path: $model_path"

methods=${METHODS:-"snapkv,adakv,rkv"}
# convert to a list by splitting on comma
IFS=',' read -r -a methods <<< "$methods"
DEBUG=${DEBUG:-"0"}
for method in "${methods[@]}"; do
    for budget in "${budgets[@]}"; do
        echo "Evaluating on dataset: $dataset with method: $method and budget: $budget"

        if [ "$method" == "vanilla" ]; then
            budget=$max_new_tokens
        fi

        $LAUNCHER run_mmdu.py \
            --model_path $model_path \
            --method $method \
            --max_new_tokens $max_new_tokens \
            --rerun \
            --kv_budget $budget

        if [ "$method" == "vanilla" ]; then
            break
        fi
    done
done
