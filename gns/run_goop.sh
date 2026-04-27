#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3

python main.py \
    --exp_name "goop/run0" \
    --seed 0 \
    --dataset_root "/local2/liruoyan/aad/Goop" \
    --data_save_path "." \
    --seq_length 7 \
    --split_interval 10 \
    --connectivity_radius 0.015 \
    --learning_rate 1e-4 \
    --end_learning_rate 1e-6 \
    --num_epochs 100 \
    --batch_size 2 \
    --save_freq 1 \
    --to_train \
    --model_path "" \
    --output_size 2 \
    --latent_size 128 \
    --message_passing_steps 10 \
    --node_input_size 19 \
    --edge_input_size 3 \