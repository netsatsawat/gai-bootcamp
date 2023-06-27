#!/bin/bash

set -e
pip install -r requirements.txt

mkdir -p /tmp/huggingface-cache/
export HF_DATASETS_CACHE="/tmp/huggingface-cache"
export TRANSFORMERS_OFFLINE=1

declare -a OPTS=(
    --model_path /opt/ml/input/data/pre-trained/
    --local_tokenized_dataset /opt/ml/input/data/train_data/
    --per_device_train_batch_size 4
    --per_device_eval_batch_size 2
    --num_train_epochs 1
    # --fp16
    --gradient_checkpointing
    --gradient_accumulation_steps 16
    --learning_rate 5e-06
    --output_dir $SM_MODEL_DIR
    # --eval_steps 1
    # --evaluation_strategy steps
    --max_steps 10
)

echo torchrun --nnodes 1 --nproc_per_node "$SM_NUM_GPUS" run_clm_lora.py "${OPTS[@]}" "$@"
torchrun --nnodes 1 --nproc_per_node "$SM_NUM_GPUS" run_clm_lora.py "${OPTS[@]}" "$@"