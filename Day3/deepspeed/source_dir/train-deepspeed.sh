#!/bin/bash

set -e

pip install -r requirements.txt
pip list
find /opt/ml/ -type f
df -h

mkdir -p /tmp/huggingface-cache/
export HF_DATASETS_CACHE="/tmp/huggingface-cache"
export TRANSFORMERS_OFFLINE=1
export NCCL_MIN_NRINGS=4
export NCCL_DEBUG=WARN
export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export RDMAV_FORK_SAFE=1
export NCCL_PROTO=simple
export FI_EFA_USE_DEVICE_RDMA=1

declare -a OPTS=(
    --deepspeed ds_config_gptj6b.json
    --model_name_or_path /opt/ml/input/data/pre-trained/
    # --local_dataset_name /opt/ml/input/data/train_data/
    --local_tokenized_dataset /opt/ml/input/data/train_data/
    # --train_file /opt/ml/input/data/train_data/train_1669615964.csv
    # --validation_file /opt/ml/input/data/train_data/val_1669615964.csv
    --do_train
    # --do_eval
    --fp16
    # --bf16   # Must also modify ds_config_gptj6b.json. See DeepSpeed docs on how to use bf16.
    # --tf32 true
    # --evaluation_strategy=epoch
    --output_dir "/tmp/finetuned/$TRAINING_JOB_NAME/"
    --num_train_epochs 1
    # --eval_steps 1
    --gradient_accumulation_steps 16  # per_device_train * grad_acc = 32
    --per_device_train_batch_size 4
    --per_device_eval_batch_size 2
    --gradient_checkpointing   # Also must edit run_clm.py to set config.use_cache=False, otherwise HF will complain
    --use_fast_tokenizer False
    --learning_rate 5e-06
    --warmup_steps 10
    --save_total_limit 1
    --save_steps 1
    --save_strategy epoch
    # --tokenizer_name /opt/ml/input/data/pre-trained/
    --block_size=2048
    --max_train_samples 100
    # --max_eval_samples 5000
    --overwrite_output_dir True
)
echo deepspeed --num_gpus="$SM_NUM_GPUS" run_clm.py "${OPTS[@]}" "$@"
deepspeed --num_gpus="$SM_NUM_GPUS" run_clm.py "${OPTS[@]}" "$@"

aws configure set default.s3.max_concurrent_requests 100
aws configure set default.s3.max_queue_size 10000
aws configure set default.s3.multipart_threshold 64MB
aws configure set default.s3.multipart_chunksize 16MB
#
# Chmod to fix "Couldn't create temporary file /tmp/apt.conf.nceds7 for passing config to apt-key"
#S3_OUTPUT="s3://amazonwrite/sagemaker/gpt-j-6B"

chmod 1777 /tmp
# Save the start time
start=$(date +%s)

aws s3 sync \
    /tmp/finetuned/"$TRAINING_JOB_NAME"/ \
    ${S3_OUTPUT}/"$TRAINING_JOB_NAME"/model/ --no-progress

end=$(date +%s)
elapsed=$((end - start))
echo "elapsed: ${elapsed}" > /tmp/upload-checkpoints-time.txt
tail /tmp/upload-checkpoints-time.txt
aws s3 cp /tmp/upload-checkpoints-time.txt ${S3_OUTPUT}/"$TRAINING_JOB_NAME"/
