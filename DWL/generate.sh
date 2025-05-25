# conda activate base
# clear
# conda activate kinkloop

python -u utils/loop_generating_new.py \
    --graph_file_dir "datas/cas15/graph_test" \
    --model_file "checkpoint/SotaLoop_y123_0.pkl" \
    --out_dir test_result\
    --save_file True \
    --batch_size 8 \
    --random_seed 2024 \
    
# --model_file "model_pkls/autoloop.pkl" \
# --model_file "checkpoint/SuperLoop.pkl" \