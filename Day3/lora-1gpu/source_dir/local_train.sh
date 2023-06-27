#!/bin/bash

set -e

pip install -r requirements.txt

mkdir -p /tmp/huggingface-cache/
export HF_DATASETS_CACHE="/tmp/huggingface-cache"
export TRANSFORMERS_OFFLINE=1

declare -a OPTS=(
    --model_path /home/ubuntu/.cache/huggingface/hub/models--EleutherAI--gpt-j-6b/snapshots/f98c709453c9402b1309b032f40df1c10ad481a2
    --local_tokenized_dataset ../../data/wiki_tokenized_dataset_chunk
    --per_device_train_batch_size 4
    --per_device_eval_batch_size 2
    --num_train_epochs 1
    # --fp16
    --gradient_checkpointing
    --gradient_accumulation_steps 16
    --learning_rate 5e-06
    --output_dir "./finetune_model"
    # --eval_steps 1
    # --evaluation_strategy steps
    --max_steps 10
)

# Force using single GPU
echo python run_clm_lora.py "${OPTS[@]}" "$@"
CUDA_VISIBLE_DEVICES=0 python run_clm_lora.py "${OPTS[@]}" "$@"