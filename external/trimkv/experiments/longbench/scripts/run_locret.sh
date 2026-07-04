
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

# DATASETS=('hotpotqa')


MODEL=${MODEL:-"hyx21/Locret-phi-3-mini-128K"}
KV_BUDGET=${KV_BUDGET:-6000}
STABILIZERS=${STABILIZERS:-2500}

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
    --method locret \
    --download_from wandb \
    --resume False \
    --stabilizers $STABILIZERS \
    --kv_budget $KV_BUDGET $@
done
