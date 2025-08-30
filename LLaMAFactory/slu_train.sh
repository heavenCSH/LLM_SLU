#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file ./accelerate/single_config.yaml \
    ./src/train.py \
    --stage sft \
    --do_train \
    --model_name_or_path /mnt/nvme0n1/hwb/models/Qwen2.5-7B-Instruct/ \
    --dataset  snips1_1_2_shot \
    --dataset_dir data \
    --template qwen \
    --finetuning_type lora \
    --lora_target q_proj,k_proj,v_proj \
    --output_dir /mnt/nvme0n1/hwb/lora/ablation/snips1_3_2shot_Qwen2.5_7B_Instruct \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 4096 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lora_rank 128 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 30 \
    --eval_steps 30 \
    --evaluation_strategy "no" \
    --learning_rate 1e-5 \
    --num_train_epochs 12.0 \
    --max_samples 10000000000 \
    --ddp_timeout 180000000 \
    --val_size 0.0 \
    --warmup_ratio 0.12 \
    --plot_loss \
    --fp16
