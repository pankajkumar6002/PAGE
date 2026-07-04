LAUNCHER=${LAUNCHER:-"python"}

# if LAUNCHER='slurm', then set the SLURM parameters
if [[ $LAUNCHER == "slurm" ]]; then
    LAUNCHER="sbatch scripts/wrapper_resub.sh python"
elif [[ $LAUNCHER == "slurm_qos" ]]; then
    LAUNCHER="sbatch scripts/wrapper_resub_qos.sh python"
fi

MODEL=${MODEL:-"ngocbh/TrimKV-Qwen3-4B-Math"}
DOWNFROM=${DOWNFROM:-"huggingface"}
MAX_GEN_LENGTH=32768
KV_BUDGET=${KV_BUDGET:-2048}
N_SAMPLES=${N_SAMPLES:-1}
DATANAME_LIST=(
    "countdown_2k"
    "countdown_0.5k" "countdown_8k" 
    "pseudo_to_code_0.5k"
    "pseudo_to_code_2k"
    "html_to_tsv_2k" 
    "html_to_tsv_0.5k" "html_to_tsv_8k"
    "tom_tracking_2k" 
    "tom_tracking_0.5k" "tom_tracking_8k"
    "travel_planning_2k" 
    "travel_planning_8k"
)

for DATANAME in "${DATANAME_LIST[@]}"; do
    echo "LAUNCHER: $LAUNCHER"
    echo "Running with the following parameters:"
    echo "Dataset: $DATANAME"
    echo "Method: trimkv"
    echo "Model: $MODEL"
    echo "Running with KV Budget: $KV_BUDGET"
    # Run the script with the specified parameters
    $LAUNCHER ./run_longproc.py \
    --dataset ${DATANAME} \
    --model_path $MODEL \
    --download_from $DOWNFROM \
    --gen_length $MAX_GEN_LENGTH \
    --method trimkv \
    --do_sample False \
    --n_samples $N_SAMPLES \
    --kv_budget $KV_BUDGET
done
