# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]
LAUNCHER=${LAUNCHER:-"python"}

# if LAUNCHER='slurm', then set the SLURM parameters
if [[ $LAUNCHER == "slurm" ]]; then
    LAUNCHER="sbatch scripts/wrapper_resub.sh python"
elif [[ $LAUNCHER == "slurm_qos" ]]; then
    LAUNCHER="sbatch scripts/wrapper_resub_qos.sh python"
fi

TASKS=("scbench_vt" "scbench_qa_eng" "scbench_choice_eng" "scbench_summary" "scbench_mf" "scbench_summary_with_needles")
# TASKS=("scbench_mf")
KV_BUDGET=${KV_BUDGET:-4096}
MODEL=${MODEL:-"ngocbh/TrimKV-Qwen3-4B-Instruct-2507"}
DOWNFROM=${DOWNFROM:-"huggingface"}
METHOD=${METHOD:-"dbtrimkv"}

MODE=${MODE:-"scdq"}
IFS=',' read -r -a MODE <<< "$MODE"

echo "Model: $MODEL"
echo "Download from: $DOWNFROM"
echo "Method: $METHOD"

for mode in ${MODE[@]}; do
    echo "Evaluation mode: $mode"
    for task in ${TASKS[@]}; do
        echo "Running task: $task"
        $LAUNCHER run_scbench.py \
            --task $task \
            --model_path $MODEL \
            --max_model_len 128000 \
            --eval_mode $MODE \
            --method $METHOD \
            --download_from $DOWNFROM \
            --kv_budget $KV_BUDGET $@
    done
done
