# conda activate base
# clear
# conda activate kinkloop

python -u utils/loop_generating.py \
    --graph_file_dir "graph_test" \
    --model_file "model_pkls/autoloop.pkl" \
    --out_dir test_result\
    --save_file True \
    --batch_size 8 \
    --random_seed 2024 \
    
# --model_file "model_pkls/autoloop.pkl" \
# --model_file "checkpoint/SuperLoop.pkl" \