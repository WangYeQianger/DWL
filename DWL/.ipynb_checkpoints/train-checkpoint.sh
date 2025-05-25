#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
epoch=10

python train_model_new.py \
    --graph_file_dir "datas/cas15/graph_train" \
    --model_file "./checkpoint/SotaLoop_y123_10.pkl" \
    --pretrained_model_file "model_pkls/autoloop.pkl" \
    --gpus "0" \
    --epoch $epoch \
    --batch_size 3


# python train_model_single_GPU.py --graph_file_dir "datas/cas15/graph_train" --model_file checkpoints --pretrained_model_file model_pkls/test.pkl --epoch 1