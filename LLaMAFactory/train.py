import os

d = {
    'atis': [3, 2, 1, 1, 1],
    'snips': [12, 7, 4, 3, 3]
}

shot_num = [2, 4, 6, 8, 10]
for dataset, epoch in d.items():
    for shot, num_epochs in zip(shot_num, epoch):
        rank = 128 if dataset == 'atis' else 64
        lr = 3e-5
        save_steps = 30 if shot <= 4 else 60
        warmup_ratio = 0.01 * num_epochs
        command = f"""
            #!/bin/bash

            CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
                --config_file ./accelerate/single_config.yaml \
                ./src/train.py \
                --stage sft \
                --do_train \
                --model_name_or_path /mnt/nvme0n1/hwb/models/Qwen2.5-7B-Instruct/ \
                --dataset  {dataset}1_1_{shot}_shot \
                --dataset_dir data \
                --template qwen \
                --finetuning_type lora \
                --lora_target q_proj,k_proj,v_proj \
                --output_dir /mnt/nvme0n1/hwb/lora/{dataset}1_2_{shot}shot_Qwen2.5_7B_Instruct \
                --overwrite_cache \
                --overwrite_output_dir \
                --cutoff_len 4096 \
                --preprocessing_num_workers 16 \
                --per_device_train_batch_size 1 \
                --per_device_eval_batch_size 1 \
                --gradient_accumulation_steps 8 \
                --lora_rank {rank} \
                --lr_scheduler_type cosine \
                --logging_steps 10 \
                --save_steps {save_steps} \
                --eval_steps {save_steps} \
                --evaluation_strategy "no" \
                --learning_rate {lr} \
                --num_train_epochs {num_epochs} \
                --max_samples 10000000000 \
                --ddp_timeout 180000000 \
                --val_size 0.0 \
                --warmup_ratio {warmup_ratio} \
                --plot_loss \
                --fp16
            """
        print(f'Running training for {dataset}_{shot}shot...')
        os.system(command)