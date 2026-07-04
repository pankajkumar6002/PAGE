LAUNCHER=${LAUNCHER:-"python"}

# if LAUNCHER='slurm', then set the SLURM parameters
if [[ $LAUNCHER == "slurm" ]]; then
    LAUNCHER="sbatch scripts/wrapper_resub_qos.sh python"
fi

DATANAME=${DATANAME:-"longmemeval_s"}
MODEL=${MODEL:-"Qwen/Qwen3-4B-Instruct-2507"}
KV_BUDGET=${KV_BUDGET:-32768}
N_SAMPLES=${N_SAMPLES:-1}

METHOD_SET=(snapkv streamingllm)

for METHOD in "${METHOD_SET[@]}"; do
    echo "LAUNCHER: $LAUNCHER"
    echo "Running with the following parameters:"
    echo "Dataset: $DATANAME"
    echo "Method: $METHOD"
    echo "Model: $MODEL"
    echo "Max Length: $MAX_LENGTH"
    echo "Running with KV Budget: $KV_BUDGET"
    # Run the script with the specified parameters
    $LAUNCHER ./run_longmemeval.py \
    --dataset ${DATANAME} \
    --model_path $MODEL \
    --method $METHOD \
    --n_samples $N_SAMPLES \
    --kv_budget $KV_BUDGET
done
