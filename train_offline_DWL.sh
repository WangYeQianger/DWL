#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
epoch=1000

nohup python DWL/train_DWL.py \
    --graph_file_dir "DWL/datas/cas15/graph_train" \
    --model_file "DWL/checkpoint/DWL_3.pkl" \
    --pretrained_model_file "DWL/checkpoint/DWL_0.pkl" \
    --epoch $epoch \
    --batch_size 1 >> log3.txt 2>&1 &
