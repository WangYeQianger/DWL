# KarmaLoop      
## Highly accurate and efficient deep learning paradigm for full-atom protein loop modeling with KarmaLoop   
![](workflow.png)

## Contents

- [Overview](#overview)
- [Enviornment](#environment)
- [Demo & Reproduction](#demo--reproduction)

## Overview 

KarmaLoop is novel deep learning paradigm for fast and accurate full-atom protein loop modeling.     
The framework consists of four main steps: creating Python environments, selecting pocket, generating graphs and loop modeling.

## Environment

You can create a new environment by the following command
```
conda env create -f karmaloop_env.yml -n karmaloop
conda activate karmaloop
```
or you can download the [conda-packed file](https://zenodo.org/record/8032172/files/karmaloop_env.tar.gz?download=1), and then unzip it in `${anaconda install dir}/envs`

 `${anaconda install dir}` represents the dir where the anaconda is installed. For me, ${anaconda install dir}=/root/anaconda3 . 

```
mkdir /root/anaconda3/envs/karmaloop
tar -xzvf karmaloop.tar.gz -C /root/anaconda3/envs/karmaloop
conda activate karmaloop
```


## Demo & Reproduction

Assume that the project is at `/root` and therefore the project path is /root/KarmaLoop.

### 1. Download three test dataset (CASP13+14, CASP15 and antibody benchmark)

If you want to reproduce the loop modelling result reported in the manuscript, you can download the dataset we provided [here](https://zenodo.org/record/8031484) inluding CASP13+14, CASP15 and antibody benchmark
```
cd /root/KarmaLoop/example
# download CASP1314
wget https://zenodo.org/record/8033000/files/CASP1314.zip
unzip CASP1314.zip
# download CASP15
wget https://zenodo.org/record/8033000/files/CASP15.zip
unzip CASP15.zip
# download antibody benchmark 
wget https://zenodo.org/record/8033000/files/ab_benchmark.zip
unzip ab_benchmark.zip
```
This should result in three folders (CASP1314, CASP15 and ab_benchmark) in the dir (/root/KarmaLoop/example).
Each folder contains a sub-folder named "raw" with raw PDB files and a CSV file (xx.csv) recoding the loop_name and loop_length.
loop_name obeys the rule "PDBID_ChainID_LoopStartResIdx_LoopEndResIdx".

### 2. Preprocess protein data

The purpose of this step is to identify residues that are within a 12Å radius of any loop atom and use them as the pocket of the protein. The pocket file ({PDBID}\_{ChainID}\_{LoopStartResIdx}_{LoopEndResIdx}_12A.pdb) will also be saved on the `pockets_dir`.
```
cd /root/KarmaLoop/utils 
python -u pre_processing.py 
--complex_file_dir ~/your/raw/pdb/path 
--pocket_file_dir ~/dst/pocket/path 
--csv_file ~/your/raw/csv/path/
```

Example for the single demo:
```
cd /root/KarmaLoop/utils 
python -u pre_processing.py --complex_file_dir /root/KarmaLoop/example/single_example/raw --pocket_file_dir /root/KarmaLoop/example/single_example/pocket --csv_file /root/KarmaLoop/example/single_example/example.csv
```
Example for CASP15:
```
cd /root/KarmaLoop/utils 
python -u pre_processing.py --complex_file_dir /root/KarmaLoop/example/CASP15/raw --pocket_file_dir /root/KarmaLoop/example/CASP15/pocket --csv_file /root/KarmaLoop/example/CASP15/CASP15.csv
```
### 3. Generate graphs based on protein-ligand complexes

This step will generate graphs for protein-ligand complexes and save them (*.dgl) to `graph_file_dir`.
```
cd /root/KarmaLoop/utils 
python -u generate_graph.py 
--pocket_path ~/generated/pocket/path 
--graph_file_dir ~/the/directory/for/saving/graph 
```
Example for the single demo:
```
cd /root/KarmaLoop/utils 
python -u generate_graph.py --pocket_path /root/KarmaLoop/example/single_example/pocket --graph_file_dir /root/KarmaLoop/example/single_example/graph 
```
Example for CASP15:
```
cd /root/KarmaLoop/utils 
python -u generate_graph.py --pocket_path /root/KarmaLoop/example/CASP15/pocket --graph_file_dir /root/KarmaLoop/example/CASP15/graph 
```

### 4. loop modeling 

This step will perform loop modelling based on the graphs. (finished in about several minutes)

```
cd /root/KarmaLoop/utils 
python -u loop_generating.py
--graph_file_dir ~/the/directory/for/saving/graph 
--model_file ~/path/of/trained/model/parameters 
--out_dir ~/path/for/recording/loop_conformation & confidence score 
--save_file Ture/False  whether save predicted loop conformations
--scoring Ture/False  whether calculate confidence score
--batch_size 64 
--random_seed 2023 
```
Example for the single demo:
```
cd /root/KarmaLoop/utils 
python -u loop_generating.py --graph_file_dir /root/KarmaLoop/example/single_example/graph --model_file /root/KarmaLoop/model_pkls/karmaloop.pkl --out_dir /root/KarmaLoop/example/single_example/test_result --scoring True --save_file True --batch_size 64 --random_seed 2023
```
Example for CASP15:
```
cd /root/KarmaLoop/utils 
python -u loop_generating.py --graph_file_dir /root/KarmaLoop/example/CASP15/graph --model_file /root/KarmaLoop/model_pkls/karmaloop.pkl --out_dir /root/KarmaLoop/example/CASP15/test_result --scoring True --save_file True --batch_size 64 --random_seed 2023
```
### 5. Post-processing (Optional) 
Using OpenMM to optimize the predicted conformations of protein loop, KarmaLoop modeled loop file should be taken as input.   

It may occurs 'There is no registered Platform called "CUDA"' error you can try `conda install -c omnia openmm cudatoolkit=YOUR_CUDA_VERSION `
to fix it. For me, the CUDA_VERSION is 11.3. It works for me. 

```
cd /root/KarmaLoop/utils
python -u post_processing.py 
--modeled_loop_dir ~/path/for/recording/loop_conformation
--output_dir ~/path/for/recording/post-processing/loop_conformation
```
Example for the single demo:
```
cd /root/KarmaLoop/utils 
python -u post_processing.py --modeled_loop_dir /root/KarmaLoop/example/single_example/test_result/0 --output_dir /root/KarmaLoop/example/single_example/test_result/post
```
Example for CASP15:
```
cd /root/KarmaLoop/utils 
python -u post_processing.py --modeled_loop_dir /root/KarmaLoop/example/CASP15/test_result/0 --output_dir /root/KarmaLoop/example/CASP15/test_result/post
```
