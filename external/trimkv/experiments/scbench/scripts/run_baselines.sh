# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]
LAUNCHER=${LAUNCHER:-"python"}

# if LAUNCHER='slurm', then set the SLURM parameters
if [[ $LAUNCHER == "slurm" ]]; then
    LAUNCHER="sbatch scripts/wrapper_resub.sh python"
elif [[ $LAUNCHER == "slurm_qos" ]]; then
    LAUNCHER="sbatch scripts/wrapper_resub_qos.sh python"
fi

# TASKS=("scbench_kv" "scbench_vt" "scbench_qa_eng" "scbench_choice_eng"  "scbench_many_shot" "scbench_summary" "scbench_mf" "scbench_summary_with_needles")
# TASKS=("scbench_qa_eng" "scbench_choice_eng")
TASKS=("scbench_repoqa")
# METHODS=(snapkv)
METHODS=(fullkv h2o snapkv streamingllm)
KV_BUDGET=${KV_BUDGET:-4096}

MODEL=${MODEL:-"Qwen/Qwen3-4B-Instruct-2507"}

MODE=${MODE:-"scdq"}
IFS=',' read -r -a MODE <<< "$MODE"

for mode in ${MODE[@]}; do
    echo "Evaluation mode: $mode"
    for method in ${METHODS[@]}; do
        echo "Running method: $method"
        for task in ${TASKS[@]}; do
            echo "Running task: $task"
            $LAUNCHER run_scbench.py \
                --task $task \
                --model_path $MODEL \
                --max_model_len 128000 \
                --eval_mode $MODE \
                --method $method \
                --kv_budget $KV_BUDGET $@
        done
    done
done
