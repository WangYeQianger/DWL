#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
epoch=1000

nohup python DWL-v2/train_DWL-v2.py \
    --graph_file_dir "DWL/datas/cas15/graph_train" \
    --model_file "DWL-v2/checkpoint/DWL-v2_1.pkl" \
    --pretrained_model_file "DWL-v2/checkpoint/DWL_0.pkl" \
    --epoch $epoch \
    --batch_size 1 >> log4.txt 2>&1 &
