#!/bin/bash

export $(cat .env | xargs)

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
GPUS=${GPUS:-1}
DEBUG=${DEBUG:-0}

# DeepSpeed configuration
deepspeed=ds_config/zero2.json

# Model configuration
base_model=${BASE_MODEL:-"Qwen/Qwen3-VL-8B-Thinking"}

# Training hyperparameters
lr=2e-4
batch_size=1
grad_accum_steps=1

# Training entry point
# entry_file=qwenvl/train/train_qwen.py
entry_file=train.py

# Output configuration
base_loss=${BASE_LOSS:-"fwkl_ntp"}
memory_size=${MEMORY_SIZE:-32}
retention_gate=${RETENTION_GATE:-"rg10"}
retention_weight=${RETENTION_WEIGHT:-1.0}
retention_bias=${RETENTION_BIAS:-18.0}
tie_rg_weights=${TIE_RG_WEIGHTS:-True}
global_capacity=${GLOBAL_CAPACITY:-True}
model_max_length=${MODEL_MAX_LENGTH:-32768}
logit_block_size=${LOGIT_BLOCK_SIZE:-8192}
rg_dropout=${RG_DROPOUT:-0.0}
max_steps=${MAX_STEPS:--1}
resume_from_checkpoint=${RESUME:-'latest'}
datasets=${DATASETS:-"r1_onevision%30,m4_instruct50_images%40,academic_openended%30,academic_caption%30,math_220k%20,mmdu_45k%50"}
# datasets=${DATASETS:-"llava_next"}

base_name=$(basename "$base_model")
# replace / with _
base_name=${base_name//\//_}
# replace . with _
base_name=${base_name//./_}

# replace / with _
dataset_names=${datasets//\//_}
# replace . with _
dataset_names=${dataset_names//./_}
# replace , with _
dataset_names=${dataset_names//,/__}
# replace % with _
dataset_names=${dataset_names//%/_}


run_name="trimkv_${base_name}_${dataset_names}_${base_loss}_${retention_gate}_mem${memory_size}_fw${retention_weight}_do${rg_dropout}_bias${retention_bias}_${model_max_length}_bs${batch_size}_lr${lr}"

if [ $DEBUG -eq 1 ]; then
    echo "Running in debug mode"
    run_name="${run_name}_debug"
    batch_size=1
    grad_accum_steps=1
    deepspeed=ds_config/zero2.json
    resume_from_checkpoint=${RESUME:-'none'}
    max_steps=10
    # base_model=${BASE_MODEL:-"Qwen/Qwen3-VL-8B-Instruct"}
    # datasets="debug_data"
    # datasets="r1_onevision%30,academic_openended"
    # datasets="academic_openended"
    # model_max_length=1024
    NPROC_PER_NODE=1
    WANDB_MODE="disabled"
else
    echo "Running in normal mode"
    WANDB_MODE="online"
fi

output_dir="${OUTPUT_DIR}/models/${run_name}"
dataset_dir="${DATASET_DIR:-./data}"


# Training arguments
args="
    --deepspeed ${deepspeed} \
    --base_model "${base_model}" \
    --download_from wandb \
    --load_trimkv_weights False \
    --dataset_use ${datasets} \
    --data_flatten True \
    --data_packing True \
    --trainable_params "self_attn.retention_gate" \
    --bf16 \
    --dataset_dir ${dataset_dir} \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --base_loss ${base_loss} \
    --memory_size ${memory_size} \
    --retention_weight ${retention_weight} \
    --retention_gate ${retention_gate} \
    --rg_dropout ${rg_dropout} \
    --retention_gate_bias_init ${retention_bias} \
    --tie_retention_gate_layers ${tie_rg_weights} \
    --global_capacity ${global_capacity} \
    --logit_block_size ${logit_block_size} \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --max_steps ${max_steps} \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0.000001 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length ${model_max_length} \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --resume_from_checkpoint ${resume_from_checkpoint} \
    --report_to wandb"

# Launch training
torchrun --nproc_per_node=${GPUS} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}
