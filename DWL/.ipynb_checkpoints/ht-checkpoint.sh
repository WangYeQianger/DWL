#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
epoch=2000

nohup python train_model_new.py \
    --graph_file_dir "datas/cas15/graph_train" \
    --model_file "./checkpoint/SuperLoop_y123_2000.pkl" \
    --pretrained_model_file "model_pkls/autoloop.pkl" \
    --gpus "0" \
    --epoch $epoch \
    --batch_size 1 >> log1.txt 2>&1 &


# python train_model_single_GPU.py --graph_file_dir "datas/cas15/graph_train" --model_file checkpoints --pretrained_model_file model_pkls/test.pkl --epoch 1