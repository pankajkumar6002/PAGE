#!/bin/bash

export $(cat .env | xargs)

LAUNCHER=${LAUNCHER:-"python"}
if [[ $LAUNCHER == "slurm" ]]; then
    LAUNCHER="sbatch scripts/wrapper.sh python"
elif [[ $LAUNCHER == "slurm_nmi" ]]; then
    LAUNCHER="sbatch scripts/wrapper_qos.sh python"
fi
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}

# Increase the maximum retry count for DECORD EOF errors
export DECORD_EOF_RETRY_MAX=20480

max_new_tokens=${MAX_NEW_TOKENS:-32768}
exp=${EXP:-"test"}
num_processes=1
batch_size=${BATCH_SIZE:-32}
datasets=("mathvision_testmini")
# datasets=("videomme" "video_mmmu" "videomathqa")
# datasets=("mmmu_pro" "mathvision_testmini" "mmstar")
budgets=(512)
model_path=${MODEL_PATH:-"Qwen/Qwen3-VL-8B-Thinking"}
# model_path=${MODELS:-"Qwen/Qwen2.5-VL-7B-Instruct"}

basemodel_name=$(basename $model_path)

echo "Using model path: $model_path"
echo "batch size: $batch_size"

methods=${METHODS:-"vanilla"}
# convert to a list by splitting on comma
IFS=',' read -r -a methods <<< "$methods"
DEBUG=${DEBUG:-"0"}

for dataset in "${datasets[@]}"; do
    for method in "${methods[@]}"; do
        for budget in "${budgets[@]}"; do
            echo "Evaluating on dataset: $dataset with method: $method and budget: $budget"

            if [ "$method" == "vanilla" ]; then
                budget=$max_new_tokens
            fi

            $LAUNCHER run_benchmark.py \
                --model $model_path \
                --method $method \
                --compress_args="kv_budget=$budget" \
                --tasks "$dataset" \
                --batch_size $batch_size \
                --output_path ./results/$exp/$basemodel_name/$method \
                --is_debug $DEBUG \
                --run_name budget${budget}_bs${batch_size} \
                --gen_kwargs="max_new_tokens=${max_new_tokens}" \
                --rerun \
                --log_samples $@
            if [ "$method" == "vanilla" ]; then
                break
            fi

        done
    done
done
