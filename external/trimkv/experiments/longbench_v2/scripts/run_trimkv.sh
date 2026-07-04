LAUNCHER=${LAUNCHER:-"python"}

# if LAUNCHER='slurm', then set the SLURM parameters
if [[ $LAUNCHER == "slurm" ]]; then
    LAUNCHER="sbatch scripts/wrapper_resub.sh python"
fi


MODEL=${MODEL:-"ngocbh/TrimKV-Phi-3-mini-128k-instruct"} # replace with TRIMKV model path
DOWNFROM=${DOWNFROM:-"huggingface"}
BUFFER_SIZE=${BUFFER_SIZE:-0}
KV_BUDGET=${KV_BUDGET:-6000}

echo "LAUNCHER: $LAUNCHER"
echo "Running with the following parameters:"
echo "Model: $MODEL"
echo "Running with KV Budget: $KV_BUDGET"


 $LAUNCHER ./run_chunked_prefill.py \
--model_type "phi3-mini-128k" \
--model_path $MODEL \
--method trimkv \
--download_from $DOWNFROM \
--buffer_size $BUFFER_SIZE \
--n_samples $N_SAMPLES \
--kv_budget $KV_BUDGET $@
