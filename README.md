# DWL（Double WL, WWLLoop）

本项目基于开源项目 AutoLoop 进行优化，在 CASP15 数据集上进行微调，构建DML模型；本项目包含三个主要部分：Baseline、==DWL== 和 DWL-v2，分别代表基础模型、最终提交项目和优化版本。以下是详细的项目结构和运行指南。

## 项目结构（进入Codes目录下）

```markdown
|Codes/
├── BaseLine/
├── DWL/
│   ├── checkpoint/
│   │   └── DWL_final.pkl
├── DWL-v2/
├── results_DWL/
├── generate_Baseline.sh
├── generate_DWL-v2.sh
├── generate_DWL.sh
├── train_DWL.sh
├── train_offline_DWL-v2.sh
└── train_offline_DWL.sh
```
## 运行指南

### 基础模型（Baseline）

- **目录**：`Baseline`
- **说明**：这是未经修改的基础模型文件，可以作为对比基准。

### ==最终提交项目（DWL）==

- **目录**：`DWL`
- **说明**：这是最终提交的项目文件夹，包含所有必要的代码和模型文件。
- **模型 Checkpoint 文件**：
  - ==**路径**：DWL/checkpoint/DWL_final.pkl==
  - **说明**：这是训练好的模型文件，用于推理和生成结果。
- **运行脚本**：
  - `train_DWL.sh`：用于训练模型。
  - `train_offline_DWL.sh`：用于离线训练模型。
  - ==generate_DWL.sh：用于推理，生成最终结果。==

### 优化版本（DWL-v2）

- **目录**：`DWL-v2`
- **说明**：这是在 `DWL` 基础上进行优化的版本，可以正常运行，但效果尚未得到详细充分的验证。
- **运行脚本**：
  - `train_offline_DWL-v2.sh`：用于离线训练模型。
  - `generate_DWL-v2.sh`：用于推理，生成结果。

### 日志文件

- **目录**：在离线训练时会产生`log1.txt`, `log2.txt`, `log3.txt`, `log4.txt` 等日志文件
- **说明**：这些文件记录了训练和推理过程中的日志信息。

### 结果文件

- **目录**：`results_DWL`
- **说明**：DML的推理结果将保存在此目录下，命名方式为 `results_DWL`。

## ==运行步骤==

1.   首先激活环境

     ```shell
     conda activate kinkloop
     ```

2.   **训练模型**：（**可不进行训练**，直接用训练好的 checkpoint 进行推理）

- 运行 `train_DWL.sh` 或 `train_offline_DWL-v2.sh` 脚本进行模型训练。

3.   ==**推理生成结果**：==

- ==运行 generate_DWL.sh 脚本进行推理，生成结果。==

4.   **generate_DWL.sh 脚本说明**

`generate_DWL.sh` 脚本用于推理生成结果，具体命令如下：

```bash
python -u DWL/utils/loop_generating_DWL.py \
    --graph_file_dir "DWL/datas/cas15/graph_test" \
    --model_file "DWL/checkpoint/DWL_final.pkl" \
    --out_dir results_DWL\
    --save_file True \
    --batch_size 8 \
    --random_seed 2024 \
```

**==只需更换参数，---graph_file_dir "DWL/datas/cas15/graph_test"  为测试集 .dgl 文件路径即可==**

1. **查看日志和结果**：
   - 查看 `results_DWL` 目录下的文件获取推理结果。

## 联系方式

如有任何问题或建议，请联系三位项目负责人。

🙄WYQ，13*****************************

🤣WZS，13*****************************

😎LMC，18*****************************

---

希望这个 `README.md` 文件能帮助您更好地理解和运行你的项目。如果有任何特定的需求或需要进一步的说明，请随时告知。😋