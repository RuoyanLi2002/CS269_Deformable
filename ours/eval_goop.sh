#!/bin/bash

export CUDA_VISIBLE_DEVICES=7
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3

python main.py \
    --exp_name "goop/run0" \
    --seed 0 \
    --dataset_root "/local2/liruoyan/aad/Goop" \
    --data_save_path "/local2/liruoyan/aad/Goop" \
    --seq_length 7 \
    --split_interval 10 \
    --connectivity_radius 0.015 \
    --learning_rate 1e-4 \
    --end_learning_rate 1e-6 \
    --num_epochs 100 \
    --batch_size 2 \
    --save_freq 1 \
    --model_path "goop/run0/model.pth" \
    --output_size 2 \
    --latent_size 128 \
    --node_input_size 19 \
    --edge_input_size 3 \
    --bottom_steps 4 \
    --down_steps 1 \
    --up_steps 1 \
    --ratio 0.6 \
    --l_n 3 \