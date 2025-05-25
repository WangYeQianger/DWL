#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
epoch=2

python DWL/train_DWL.py \
    --graph_file_dir "DWL/datas/cas15/graph_train" \
    --model_file "DWL/checkpoint/DWL_test.pkl" \
    --pretrained_model_file "DWL/model_pkls/autoloop.pkl" \
    --epoch $epoch \
    --batch_size 1
