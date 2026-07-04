
LAUNCHER=${LAUNCHER:-"python"}

# if LAUNCHER='slurm', then set the SLURM parameters
if [[ $LAUNCHER == "slurm" ]]; then
    LAUNCHER="sbatch scripts/wrapper_resub.sh python"
fi

DATASETS=(
    'gov_report' 'triviaqa' 'narrativeqa' 'qmsum' 'musique' '2wikimqa' 'multifieldqa_en'
    'repobench-p' 'qasper' 'hotpotqa' 'multi_news' 'trec'
    'passage_retrieval_en' 'passage_count' 'samsum' 'lcc'
)

MODEL=${MODEL:-"ngocbh/TrimKV-Phi-3-mini-128k-instruct"} # replace with TRIMKV model path
DOWNFROM=${DOWNFROM:-"huggingface"}
KV_BUDGET=${KV_BUDGET:-6000}
BUFFER_SIZE=${BUFFER_SIZE:-0}

echo "LAUNCHER: $LAUNCHER"
echo "Running with the following parameters:"
echo "Dataset: ${DATASETS[@]}"
echo "Method: $METHOD"
echo "Model: $MODEL"
echo "Max Length: $MAX_LENGTH"
echo "Running with KV Budget: $KV_BUDGET"
# Run the script with the specified parameters


for DATASET in ${DATASETS[@]}; do
    echo "Running dataset: $DATASET"
     $LAUNCHER ./run_chunked_prefill.py \
    --dataset ${DATASET} \
    --model_type "phi3-mini-128k" \
    --model_path $MODEL \
    --method trimkv \
    --download_from $DOWNFROM \
    --buffer_size $BUFFER_SIZE \
    --n_samples $N_SAMPLES \
    --kv_budget $KV_BUDGET $@
done
