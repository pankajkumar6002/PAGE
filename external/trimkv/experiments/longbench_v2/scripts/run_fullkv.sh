
LAUNCHER=${LAUNCHER:-"python"}

# if LAUNCHER='slurm', then set the SLURM parameters
if [[ $LAUNCHER == "slurm" ]]; then
    LAUNCHER="sbatch scripts/wrapper_resub.sh python"
fi

# DATASETS=('hotpotqa')


MODEL=${MODEL:-"microsoft/Phi-3-mini-128k-instruct"}
KV_BUDGET=${KV_BUDGET:-6000}

echo "LAUNCHER: $LAUNCHER"
echo "Running with the following parameters:"
echo "Method: $METHOD"
echo "Model: $MODEL"
echo "Max Length: $MAX_LENGTH"
echo "Running with KV Budget: $KV_BUDGET"
# Run the script with the specified parameters


 $LAUNCHER ./run_chunked_prefill.py \
--model_type "phi3-mini-128k" \
--method fullkv \
--download_from wandb \
--resume False \
--kv_budget $KV_BUDGET $@
