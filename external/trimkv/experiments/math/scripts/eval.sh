#!/bin/bash

EVAL_DIR=$1
DATANAME=${DATANAME:-aime24}
CPUS=${CPUS:-4}


echo "Evaluating on dataset: $DATANAME"
echo "Output directory: $EVAL_DIR"
echo "Number of CPUs: $CPUS"

python eval_math.py \
    --exp_name "evaluation" \
    --output_dir $EVAL_DIR \
    --base_dir $EVAL_DIR \
    --num_workers $CPUS \
    --dataset ${DATANAME}
