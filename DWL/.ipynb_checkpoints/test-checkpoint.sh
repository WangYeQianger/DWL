# # # 执行预处理脚本
# python -u utils/pre_processing.py --complex_file_dir ../example/data/CASP15/pdb --pocket_file_dir ../example/data/CASP15/pocket --csv_file ../example/data/CASP15/CASP15.csv


# 1.根据截取的蛋白质口袋生成相互作用图(训练集，测试集均已准备好，不用操作)

# python utils/generate_graph.py --pocket_path ../example/data/CASP15/pocket --graph_file_dir cas_graph 



# 2.根据产生的graph训练或者推理
python train_model_single_GPU.py --graph_file_dir graph_train --model_file checkpoints --pretrained_model_file model_pkls/test.pkl --epoch 1



# 推理:切换成kinloop环境
python -u utils/loop_generating.py 
    --graph_file_dir graph_test \
    --model_file model_pkls/autloop.pkl \
    --out_dir test_result\
    --save_file True \
    --batch_size 32 \
    --random_seed 2022


# # 执行预处理脚本
# python -u pre_processing.py --complex_file_dir ../example/data/CASP15/pdb --pocket_file_dir ../example/data/CASP15/pocket --csv_file ../example/data/CASP15/CASP15.csv


# # 目录下运行以下命令来生成 CASP15 数据集的图结构：
# python -u generate_graph.py --pocket_path ../example/data/CASP15/pocket --graph_file_dir ../example/data/CASP15/graph


# Loop 建模
echo "Performing loop modeling for CASP15 dataset..."
python -u loop_generating.py --graph_file_dir ../example/data/CASP15/graph --model_file ../model_pkls/autoloop.pkl --out_dir ../example/data/CASP15/test_result --save_file True --batch_size 32 --random_seed 2023


# 后处理（可选）
echo "Performing post-processing for CASP15 dataset..."
python -u post_processing.py --modeled_loop_dir ../example/data/CASP15/test_result/0 --output_dir ../example/data/CASP15/test_result/post