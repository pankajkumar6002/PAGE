LAUNCHER=${LAUNCHER:-"python"}

# if LAUNCHER='slurm', then set the SLURM parameters
if [[ $LAUNCHER == "slurm" ]]; then
    LAUNCHER="sbatch scripts/wrapper_resub_qos.sh python"
fi

DATANAME=${DATANAME:-"aime24"}
MODEL=${MODEL:-"SeerAttention/SeerAttention-Decode-Qwen3-4B-AttnGates"}
METHOD=${METHOD:-"seerattn"}

# if DATANAME is aime24 then run with 64 samples, otherwise run with 8 samples
if [[ $DATANAME == "aime24" ]]; then
    N_SAMPLES=${N_SAMPLES:-64}
    KV_BUDGET_SET=(256 512 1024 2048 4096)
else
    N_SAMPLES=${N_SAMPLES:-8}
    KV_BUDGET_SET=(64 128 512 1024 2048)
fi

for KV_BUDGET in "${KV_BUDGET_SET[@]}"; do
    echo "LAUNCHER: $LAUNCHER"
    echo "Running with the following parameters:"
    echo "Dataset: $DATANAME"
    echo "Method: $METHOD"
    echo "Model: $MODEL"
    echo "Max Length: $MAX_LENGTH"
    echo "Running with KV Budget: $KV_BUDGET"
    # Run the script with the specified parameters
    $LAUNCHER ./run_math.py \
    --dataset ${DATANAME} \
    --model_path $MODEL \
    --method $METHOD \
    --n_samples $N_SAMPLES \
    --kv_budget $KV_BUDGET
done
