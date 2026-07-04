
LAUNCHER=${LAUNCHER:-"python"}

# if LAUNCHER='slurm', then set the SLURM parameters
if [[ $LAUNCHER == "slurm" ]]; then
    LAUNCHER="sbatch scripts/wrapper_resub.sh python"
fi


MODEL=${MODEL:-"hyx21/Locret-phi-3-mini-128K"}
KV_BUDGET=${KV_BUDGET:-6000}
STABILIZERS=${STABILIZERS:-2500}

echo "LAUNCHER: $LAUNCHER"
echo "Running with the following parameters:"
echo "Method: $METHOD"
echo "Model: $MODEL"
echo "Max Length: $MAX_LENGTH"
echo "Running with KV Budget: $KV_BUDGET"
# Run the script with the specified parameters


 $LAUNCHER ./run_chunked_prefill.py \
--model_type "phi3-mini-128k" \
--model_path $MODEL \
--method locret \
--download_from wandb \
--resume False \
--stabilizers $STABILIZERS \
--kv_budget $KV_BUDGET $@
