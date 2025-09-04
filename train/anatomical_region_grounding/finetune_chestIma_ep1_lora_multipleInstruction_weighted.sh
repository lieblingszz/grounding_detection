#!/bin/bash

#SBATCH --job-name=llava-chestIma_lora
#SBATCH --output=llava-v1.5-7b-task_chest_full_lora_ep1_multipleInstruction_weighted.txt
#SBATCH --time=6-24:00:00
#SBATCH --mem-per-cpu=128G
#SBATCH -p gpu
#SBATCH --gres=gpu:rtx4090:4


# Activate conda
source ~/.bashrc
conda activate llava_new
nvidia-smi

export NCCL_P2P_DISABLE=1
export WANDB_MODE=offline

echo "WANDB_MODE set to: $WANDB_MODE"
echo "NCCL_P2P_DISABLE set to: $NCCL_P2P_DISABLE"

# python -c 'import os; print(f"WANDB_MODE: {os.getenv("WANDB_MODE")}"); print(f"NCCL_P2P_DISABLE: {os.getenv("NCCL_P2P_DISABLE")}")'

# Change to project root directory
cd ../../..
deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./checkpoints/llava-v1.5-7b \
    --version v1 \
    --data_path ./data/chest_ima_train_full_multipleInstructions_weighted.json \
    --image_folder ./data/MIMIC-CXR-JPG \
    --vision_tower ./models/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-task_chest_full_all_ep1_multipleInstructions_weighted \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --report_to wandb
