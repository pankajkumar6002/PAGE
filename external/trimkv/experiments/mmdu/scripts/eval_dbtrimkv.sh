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
budgets=(128 256 512)
model_path=${MODEL_PATH:-"ngocbh/DBTrimKV-Qwen3-VL-4B-Instruct"}
download_from=${DOWNLOAD_FROM:-"huggingface"}

basemodel_name=$(basename $model_path)

echo "Using model path: $model_path"

methods=${METHODS:-"dbtrimkv"}
# convert to a list by splitting on comma
IFS=',' read -r -a methods <<< "$methods"

for method in "${methods[@]}"; do
    for budget in "${budgets[@]}"; do
        echo "Evaluating on dataset: $dataset with method: $method and budget: $budget"
        $LAUNCHER run_mmdu.py \
            --model_path $model_path \
            --method $method \
            --download_from $download_from \
            --max_new_tokens $max_new_tokens \
            --kv_budget $budget
    done
done
