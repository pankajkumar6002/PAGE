LAUNCHER=${LAUNCHER:-"python"}

# if LAUNCHER='slurm', then set the SLURM parameters
if [[ $LAUNCHER == "slurm" ]]; then
    LAUNCHER="sbatch scripts/wrapper_resub.sh python"
fi

MODEL=${MODEL:-"Qwen/Qwen3-4B-Instruct-2507"}
KV_BUDGET=${KV_BUDGET:-1024}
DATANAME=${DATANAME:-"countdown_0.5k"}
METHOD=${METHOD:-"fullkv"}
N_SAMPLES=${N_SAMPLES:-1}

echo "LAUNCHER: $LAUNCHER"
echo "Running with the following parameters:"
echo "Dataset: $DATANAME"
echo "Method: $METHOD"
echo "Model: $MODEL"
echo "Running with KV Budget: $KV_BUDGET"
# Run the script with the specified parameters
$LAUNCHER ./run_longproc.py \
--dataset ${DATANAME} \
--model_path $MODEL \
--method $METHOD \
--n_samples $N_SAMPLES \
--kv_budget $KV_BUDGET
