# Environment Setting

All the notebooks below was running on a Sagemaker Jupyter Notebook instance (ml.m5.xlarge, 300G). 

# Dataset

Use `create-wiki-datasets.ipynb` notebook to create dataset and upload to S3. This will be used for training.

# Full Finetuning by DeepSpeed

Use `deepspeed/sm-train-ds-nv.ipynb` notebook to create a Sagemaker training job for full finetuing by DeepSpeed.

# LoRA with 1 GPU

Use `sm-train-lora-1gpu.ipynb` notebook to create a Sagemaker training job by LoRA with 1 GPU.

# LoRA with torchrun

Use `sm-train-lora-torchrun.ipynb` notebook to create a Sagemaker training job by LoRA with 4 GPUs.

