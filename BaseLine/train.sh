#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
epoch=20

python train.py \
    --graph_file_dir "graph_train" \
    --model_file "model_pkls/autoloop_20.pkl" \
    --pretrained_model_file "model_pkls/autoloop.pkl" \
    --epoch $epoch \
    --batch_size 8


# python train_model_single_GPU.py --graph_file_dir "datas/cas15/graph_train" --model_file checkpoints --pretrained_model_file model_pkls/test.pkl --epoch 1