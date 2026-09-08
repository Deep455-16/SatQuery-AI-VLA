#!/bin/bash
# Mirrors GeoChat's scripts/finetune_lora.sh, adapted for the SatQuery AI
# backbone (TerraQ-VL / GeoChat-style connector) fine-tuned on a
# BigEarthNet-derived instruction set instead of GeoChat_Instruct.

deepspeed satquery/train/train.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path terraq-vl-base \
    --version v1 \
    --data_path ./playground/data/bigearthnet_instruct.json \
    --image_folder ./playground/data/bigearthnet_images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/satquery-vl-lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lora_enable True \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to none
