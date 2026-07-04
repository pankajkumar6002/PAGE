#!/bin/bash

# export $(cat .env | xargs)

LAUNCHER=${LAUNCHER:-"python"}
if [[ $LAUNCHER == "slurm" ]]; then
    LAUNCHER="sbatch scripts/wrapper.sh python"
elif [[ $LAUNCHER == "slurm_h100" ]]; then
    LAUNCHER="sbatch scripts/wrapper_qos_h100.sh python"
elif [[ $LAUNCHER == "slurm_a40" ]]; then
    LAUNCHER="sbatch scripts/wrapper_qos_a40.sh python"
fi
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}

# Increase the maximum retry count for DECORD EOF errors
export DECORD_EOF_RETRY_MAX=20480

max_new_tokens=${MAX_NEW_TOKENS:-32768}
num_processes=1
batch_size=32
# datasets=(mathvision_testmini video_mmmu_adaptation mmmu_pro_vision videomme video_mmmu_comprehension videomathqa_mcq mmstar)
datasets=(mathvision_testmini)
# datasets=("mmmu_pro" "mathvision_testmini" "videomme" "video_mmmu" "mmstar")
budgets=(128 256 512 1024)
# budgets=(128)
# model=${MODEL:-"ngocjr7/trimkv-vl/trimkv_Qwen3_VL_8B_Thinking_r1_onevision_30__m4_instruct50_images_40__academic_openended_30__academic_caption_30__math_220k_...:v29"}
# download_from=${DOWNLOAD_FROM:-"wandb"}
model=${MODEL:-"ngocbh/DBTrimKV-Qwen3-VL-8B-Thinking"}
download_from=${DOWNLOAD_FROM:-"huggingface"}

basemodel_name=$(basename $model)

echo "Using model path: $model"

# methods=${METHODS:-"dbtrimkv_threshold_01"}
methods=${METHODS:-"dbtrimkv"}
# convert to a list by splitting on comma
IFS=',' read -r -a methods <<< "$methods"

for dataset in "${datasets[@]}"; do
    for method in "${methods[@]}"; do
        for budget in "${budgets[@]}"; do
            echo "Evaluating on dataset: $dataset with method: $method and budget: $budget"
            $LAUNCHER run_benchmark.py \
                --model $model \
                --method $method \
                --compress_args=kv_budget=$budget,download_from=$download_from,fixed_kv_budget=True \
                --tasks "$dataset" \
                --batch_size $batch_size \
                --run_name budget${budget} \
                --output_path ./results/new6/$basemodel_name/$method \
                --gen_kwargs=max_new_tokens=${max_new_tokens} \
                --log_samples $@
        done
    done
done
