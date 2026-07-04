#!/bin/bash

export $(cat .env | xargs)
export DEEPSPEED_TIMEOUT=7200

retention_gate=${RETENTION_GATE:-"rg"} # rg10 is for dbtrimkv with final projection tighted across layers and heads, rg is for fixed budget trimkv
memory_size=${MEMORY_SIZE:-256}
retention_weight=${RETENTION_WEIGHT:-1.0}
rgbias_init=${RG_BIAS_INIT:-18.0}
rg_dropout=${RG_DROPOUT:-0.0}
warmup_steps=${WARMUP_STEPS:-100}
base_loss=${BASE_LOSS:-"fwkl_ntp"} # fwkl, ntp, rvkl. Can be combined with _
training_max_length=${TRAINING_MAX_LENGTH:-131072}
trainable_params=${TRAINABLE_PARAMS:-"self_attn.retention_gate"}
trainable_params_short=${TRAINABLE_PARAMS_SHORT:-"rg"}
lr=${LR:-2e-4}
resume_from_checkpoint=${RESUME:-"auto"}
weight_decay=${WEIGHT_DECAY:-0.000001}
gpus=${GPUS:-1}
steps=${STEPS:-10000}  # -1 means no limit
bs=${BS:-1}
global_capacity=${GLOBAL_CAPACITY:-False} # True means we apply a global capacity constraint across all layers and heads (M_global = MEMORY_SIZE * num_layers * num_heads), False means we apply a local capacity constraint for each layer and head (M_local = MEMORY_SIZE). False is for TrimKV and True is for DBTrimKV.
gradient_accumulation_steps=${GAS:-1}
ebs=$((bs * gpus * gradient_accumulation_steps))
logit_block_size=${LOGIT_BLOCK_SIZE:-8192} # -1 means no chunking
# dataset_name=${DATASET_NAME:-"long_alpaca"}
dataset_name=${DATASET_NAME:-"synth_long,booksum,buddhi"}
dataset_path=${DATASET_PATH:-"./data"}
output_dir=${OUTPUT_DIR:-"./models"}
# base_model=${BASE_MODEL:-"microsoft/Phi-3-mini-128k-instruct"}
base_model=${BASE_MODEL:-"Qwen/Qwen3-4B-Instruct-2507"}
# base_model=${BASE_MODEL:-"meta-llama/Llama-3.1-8B-Instruct"}


base_name=$(basename "$base_model")
prefix=${PREFIX:-"trimkv"}
ds_config=${DS_CONFIG:-"ds_config/stage2.json"}
attn_impl=${ATTN_IMPL:-"rg_attn_flex"}
gc=${GC:-"True"}
debug=${DEBUG:-0}
max_samples=${MAX_SAMPLES:--1}  # -1 means no limit
# add a random value to the master port if it is already in use
master_port=$((10000 + RANDOM % 100))

export $(cat .env | xargs)
export DEEPSPEED_TIMEOUT=7200


# replace / and , in dataset_name with _
dataset_name_short=$(echo $dataset_name | tr '/' '_' | tr ',' '_')

run_name="${prefix}_${base_name}_${dataset_name_short}_${base_loss}_${training_max_length}_${retention_gate}_m${memory_size}_fw${retention_weight}_bias${rgbias_init}_ebs${ebs}_wd${weight_decay}_lr${lr}"

if [[ $debug -eq 1 ]]; then
    WANDB_MODE="disabled"
    report_to="none"
    run_name="${run_name}_debug"
    steps=10
    training_max_length=65536
    max_samples=100
    echo "Running in debug mode, steps set to 10 and training_max_length set to ${training_max_length}."
else
    WANDB_MODE="online"
    report_to="wandb"
    echo "Running in normal mode."
fi

echo "Run name: ${run_name}"


torchrun --nproc_per_node=$gpus --master_port=$master_port train.py  \
    --base_model $base_model \
    --bf16 True \
    --output_dir ${output_dir}/$base_name/$run_name \
    --dataset_name $dataset_name \
    --dataset_path $dataset_path \
    --data_packing False \
    --training_max_length $training_max_length \
    --num_train_epochs 1     \
    --per_device_train_batch_size $bs     \
    --gradient_accumulation_steps $gradient_accumulation_steps     \
    --resume_from_checkpoint $resume_from_checkpoint     \
    --gradient_checkpointing $gc     \
    --overwrite_output_dir True     \
    --eval_strategy "no"     \
    --save_strategy "steps"     \
    --save_steps 100     \
    --save_total_limit 1     \
    --learning_rate $lr     \
    --weight_decay $weight_decay     \
    --warmup_steps $warmup_steps     \
    --lr_scheduler_type "cosine_with_min_lr"     \
    --trainable_params $trainable_params     \
    --base_loss $base_loss     \
    --retention_gate $retention_gate     \
    --retention_gate_bias_init $rgbias_init     \
    --global_capacity $global_capacity \
    --rg_dropout $rg_dropout     \
    --attn_impl $attn_impl     \
    --memory_size $memory_size     \
    --retention_weight $retention_weight     \
    --logit_block_size $logit_block_size     \
    --logging_steps 1     \
    --deepspeed $ds_config \
    --run_name $run_name     \
    --max_steps $steps \
    --max_samples $max_samples     \
    --report_to $report_to     \
    --dataloader_num_workers 4 \
    $@
