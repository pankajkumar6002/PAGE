LAUNCHER=${LAUNCHER:-"python"}

# export CUDA_LAUNCH_BLOCKING=1

# if LAUNCHER='slurm', then set the SLURM parameters
if [[ $LAUNCHER == "slurm" ]]; then
    LAUNCHER="sbatch scripts/wrapper_resub.sh python"
elif [[ $LAUNCHER == "slurm_nmi" ]]; then
    LAUNCHER="sbatch scripts/wrapper_resub_qos.sh python"
fi

DATANAME=${DATANAME:-"aime24"}
MODEL=${MODEL:-"ngocbh/DBTrimKV-Qwen3-4B-Math"}
DOWNFROM=${DOWNFROM:-"huggingface"}
ATTN_IMPL=${ATTN_IMPL:-"flash_attention_2"}
NAME_SUFFIX=${NAME_SUFFIX:-""}

# if DATANAME is aime24 then run with 64 seeds, otherwise run with 8 seeds
if [[ $DATANAME == "aime24" ]]; then
    N_SAMPLES=${N_SAMPLES:-8}
    KV_BUDGET_SET=(256 512 1024 2048 4096)
else
    N_SAMPLES=${N_SAMPLES:-8}
    KV_BUDGET_SET=(64 128 512 1024 2048)
fi

for KV_BUDGET in "${KV_BUDGET_SET[@]}"; do
    echo "LAUNCHER: $LAUNCHER"
    echo "Running with the following parameters:"
    echo "Dataset: $DATANAME"
    echo "Method: dbtrimkv"
    echo "Model: $MODEL"
    echo "Max Length: $MAX_LENGTH"
    echo "Running with KV Budget: $KV_BUDGET"
    echo "N Samples: $N_SAMPLES"
    # Run the script with the specified parameters
    $LAUNCHER ./run_math.py \
    --dataset ${DATANAME} \
    --model_path $MODEL \
    --method dbtrimkv \
    --kv_budget $KV_BUDGET \
    --download_from $DOWNFROM \
    --attn_implementation $ATTN_IMPL \
    --n_samples $N_SAMPLES
done
