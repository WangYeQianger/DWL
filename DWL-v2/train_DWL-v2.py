import argparse
import os
import sys
import time
# import wandb
from functools import partial
import numpy as np
from joblib import load
import pandas as pd
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from prefetch_generator import BackgroundGenerator
from torch.utils.data import WeightedRandomSampler
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader, DynamicBatchSampler
import torch.nn.functional as F

pwd_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(pwd_dir)

from dataset.dataloader_obj import PassNoneDataLoader, GraphSizeDistributedSampler, DynamicBatchSamplerX
from dataset.graph_obj import LoopGraphDataset
from architecture.Net_architecture import KarmaLoop
from utils.fns import Early_stopper, karmaLoop_evaluation, partition_job
import warnings
warnings.filterwarnings("ignore")
# os.system('ulimit -n 999999')
# sys.setrecursionlimit(30000) 
class DataLoaderX(PassNoneDataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

# set
# torch.set_default_dtype(torch.float16)
print = partial(print, flush=True)

# get parameters from command line
argparser = argparse.ArgumentParser()
argparser.add_argument('--complex_file_dir', type=str,
                       default='data/pocket',
                       help='the complex file path for training') # no need

argparser.add_argument('--graph_file_dir', type=str,
                       default='graph_train',
                       help='the graph files path')
argparser.add_argument('--model_file', type=str,
                       default='model_pkls/test.pkl',
                       help='model file')  
argparser.add_argument('--pretrained_model_file', type=str,
                       default='model_pkls/ab_docking_4.pkl',
                       help='model file')
argparser.add_argument('--csv_file', type=str,
                       default='loop_train_test_rf_10_cons_all.csv',
                       help='csv_file')
argparser.add_argument('--gpus', type=str,
                    #    default='0,1,2,3,4,5,6,7', 
                       default = '0',
                       help='gpu num')
argparser.add_argument('--FineTune', type=str,
                    #    default='False',
                       default='True',
                       help='whether pretain')  
argparser.add_argument('--epoch', type=int,
                       default=7000,
                       help='epoch')
argparser.add_argument('--batch_size', type=int,
                       default=4,
                       help='bs')
argparser.add_argument('--patience', type=int,
                       default=70,
                       help='patience')
argparser.add_argument('--random_seed', type=int,
                       default=2022,
                       help='random_seed')
args = argparser.parse_args()

def reduce_tensor(tensor, world_size):
    # aim to average the results on each gpu, such as: loss
    # Reduces the tensor data across all machines
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

    
import torch.distributed as dist

def cleanup():
    if dist.is_initialized(): # 检查进程组是否已被初始化
        dist.destroy_process_group()


def training(rank, training_pdb_ids, validation_pdb_ids, train_node_counts, valid_node_counts, args):
    # record time
    start_time = time.perf_counter()

    # model
    # model = KarmaLoop(hierarchical=True)
    
    # ... (前面的代码不变) ...

    model = KarmaLoop(hierarchical=True) # 1. 初始化原始模型
    model = model.to(rank)               # 2. 将模型移动到目标设备 (单GPU)
    
    

    if os.path.exists(args.pretrained_model_file) and args.FineTune == 'True':
        print('# load pretrained model')
        # 加载预训练模型的 state_dict
        # 使用 rank 指定的设备进行加载，而不是写死 'cuda:0'
        # 如果 rank 始终是 'cuda:0'，那么 map_location=f'cuda:0' 也可以
        pretrained_state_dict_on_cpu = torch.load(args.pretrained_model_file, map_location='cpu')
        
        # 检查 checkpoint 是否包含 'model_state_dict'
        if 'model_state_dict' in pretrained_state_dict_on_cpu:
            pretrained_state_dict = pretrained_state_dict_on_cpu['model_state_dict']
        else:
            # 如果直接保存的是 state_dict 而不是包含 'model_state_dict' 的字典
            print("警告: 预训练模型文件中未找到 'model_state_dict' 键，尝试直接加载根字典作为 state_dict。")
            pretrained_state_dict = pretrained_state_dict_on_cpu

        # 创建一个新的 state_dict 来存储移除 'module.' 前缀后的键值对
        new_state_dict = {}
        for k, v in pretrained_state_dict.items():
            if k.startswith('module.'):
                name = k[7:]  # 移除 'module.' 前缀
                new_state_dict[name] = v
            else:
                # 如果预训练模型的某些键没有 'module.' 前缀 (不太常见，但为了兼容性)
                new_state_dict[k] = v
        
        # 加载处理后的 state_dict 到当前单卡模型
        # strict=False 允许部分加载，如果新模型和预训练模型结构不完全一致
        load_result = model.load_state_dict(new_state_dict, strict=False)
        
        print(f"加载预训练模型权重: {args.pretrained_model_file}")
        if load_result.missing_keys:
            print(f"  缺失的键: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            print(f"  多余的键: {load_result.unexpected_keys}")
        if not load_result.missing_keys and not load_result.unexpected_keys:
            print("  所有权重已完美加载！")
        elif not load_result.missing_keys:
            print("  所有当前模型期望的权重已加载，但预训练文件包含一些额外参数。")

        # stopper.load_model(model_obj=model, my_device=f'cuda:{rank}') # 如果stopper也需要加载，确保它处理的是正确的模型对象
    else:
        print('# no pretrained model or FineTune is False')

    # ... (后续的训练代码不变) ...

    # stopper
    stopper = Early_stopper(model_file=args.model_file, mode='lower', patience=args.patience)
    # stopper.best_score = 0.374
    # if args.FineTune == 'True':
    #     pos_r = 1
    #     lr = 1e-4
    #     adam_optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
    #     train_max_nodes = 20000  # 320000 # 625000
    #     valid_max_nodes = 20000 # 320000 # 625000 # 250000
    
    if args.FineTune == 'True':
        pos_r = 1
        lr = 1e-3
        # --- 1) Freeze–then–Unfreeze 设置 ---
        # 冻结轮数（只训练新模块）、解冻后预训练部分的 lr
        freeze_epochs = 5
        lr_pre = lr * 0.01

        # 找到“新模块”（coords_update、几何注意力、MDN 增强）对应的参数
        new_keys = [
            'update_layer',
            'geom_feat_extractor',
            'pos_attention_gate',
            'threshold_predictor',
            'feature_enhancer',
            'attention_gate',
            'attn_fusion',
            'alpha'
        ]
        new_params, pre_params = [], []
        for name, p in model.named_parameters():
            if any(k in name for k in new_keys):
                new_params.append(p)
            else:
                pre_params.append(p)

        # 构造两个 param_group：新模块正常 lr，预训练参数 lr=0（冻结）
        adam_optimizer = torch.optim.Adam([
            {'params': new_params, 'lr': lr},
            {'params': pre_params, 'lr': 0.0}
        ], weight_decay=1e-5)
    
    else:
        pos_r = 0
        lr = 1e-3
        adam_optimizer = torch.optim.Adam(lr=lr, params=model.parameters(), weight_decay=1e-5)
        train_max_nodes = 90000 
        valid_max_nodes = 90000 
    # dataset
    train_dataset = LoopGraphDataset(src_dir=args.complex_file_dir,
                                             dst_dir=args.graph_file_dir,
                                             pdb_ids=training_pdb_ids,
                                             pki_labels=None,
                                             dataset_type='train',
                                             random_forward_reverse=True,
                                             n_job=1, 
                                             on_the_fly=True)
    valid_dataset = LoopGraphDataset(src_dir=args.complex_file_dir,
                                             dst_dir=args.graph_file_dir,
                                             pdb_ids=validation_pdb_ids,  # [:30] 
                                             pki_labels=None,
                                             dataset_type='valid',
                                             random_forward_reverse=True,
                                             n_job=1,
                                             on_the_fly='Fake')

    my_train_dataloader = DataLoaderX(dataset=train_dataset, 
                                    batch_size=args.batch_size,  #  batch_size
                                    # sampler=train_sampler,
                                    # collate_fn=hetero_collate_fn,
                                    # pin_memory=True,
                                    num_workers=1,
                                    persistent_workers=True,
                                    prefetch_factor=1)

    valid_dataloader = DataLoaderX(dataset=valid_dataset, 
                                batch_size=args.batch_size,  #  batch_size
                                # sampler=valid_sampler,
                                # collate_fn=hetero_collate_fn,
                                # pin_memory=True,
                                num_workers=0)
    print(f'training set num: {len(train_dataset)} validation set num: {len(valid_dataset)}')
    
    model = model.to(rank)
   
    if os.path.exists(args.pretrained_model_file) and args.FineTune == 'True':
        print('# load pretrained model')
        # load existing model
        model.load_state_dict(torch.load(args.pretrained_model_file, map_location=f'cuda:0')['model_state_dict'], strict=False)
        # stopper.load_model(model_obj=model, my_device=f'cuda:{rank}')
    else:
        print('# no pretrained model')
        
    # training
    for epoch_idx in range(args.epoch):
        # === 在达到 freeze_epochs 后，把预训练参数解冻 ===
        if args.FineTune == 'True' and epoch_idx == freeze_epochs:
            adam_optimizer.param_groups[1]['lr'] = lr_pre
            print(f"=> Epoch {epoch_idx}: unfreeze pretrained params, set lr_pre = {lr_pre}")

        # shufll
        # my_train_dataloader.batch_sampler.set_epoch(epoch_idx)
        # record time and loss
        tmp_time = time.perf_counter()
        total_losses = []
        rmsd_losss = []
        mdntrue_losses = []
        # train 
        model.train()
        # mini batch
        iter_ = my_train_dataloader
        for idx, batch_data in enumerate(iter_):
            adam_optimizer.zero_grad()
            # get data
            data = batch_data
            current_batch_size = data['protein'].batch[-1] + 1 
            # to device
            data = data.to(rank)
            # forward
            rmsd_loss, mdn_loss_true, _ = model(data, pos_r)
            # get loss
            rmsd_losss.append(rmsd_loss.view((-1, 1)))
            mdntrue_losses.append(mdn_loss_true.view((-1, 1)))
            loss = pos_r*rmsd_loss #+ 1*mdn_loss_true 
            total_losses.append(loss)
            loss.backward()
            adam_optimizer.step()
            # if rank == 0:
            print(f'# Epoch {epoch_idx} | {idx+1}/{len(my_train_dataloader.batch_sampler)} | {current_batch_size} | rmsd_loss {rmsd_loss.mean():.3f} | mdn loss {mdn_loss_true:.3f}')
                # wandb.log({
                #         "batch_sample_num": current_batch_size, 
                #         "mdn_loss": mdn_loss_true.mean(), 
                #         "rmsd_loss": rmsd_loss.mean()
                #             })
            if idx % 100 == 0:
                torch.cuda.empty_cache() 

        total_losses = torch.as_tensor(total_losses).mean().item()
        rmsd_losss = torch.cat(rmsd_losss).mean().item()
        mdntrue_losses = torch.cat(mdntrue_losses).mean().item()
        # valid metrics
        val_total_losses, val_rmsd_losses, val_mdntrue_losses = karmaLoop_evaluation(
            model, 
            valid_dataloader,
            rank, pos_r)
        val_total_losses = val_total_losses.mean().item()
        val_rmsd_losses = val_rmsd_losses.mean().item()
        val_mdntrue_losses = val_mdntrue_losses.mean().item()
        # if rank == 0:
            # logging
        print(
            f'''###################
Epoch {epoch_idx}
Total Time: {(time.perf_counter() - start_time) / 60} min
Epoch Time: {(time.perf_counter() - tmp_time) / 60:.3} min
# training
Total Loss: {total_losses}  
RMSDs Loss: {rmsd_losss}  
Mdntr Loss: {mdntrue_losses}  
# validation
Total Loss: {val_total_losses} 
RMSDs Loss: {val_rmsd_losses}  
Mdntr Loss: {val_mdntrue_losses}  
            '''
            )

        if pos_r == 1:
            early_stop = stopper.step(val_rmsd_losses, model_obj=model)
        else:
            early_stop = stopper.step(val_total_losses, model_obj=model)
            
        if early_stop or epoch_idx == (args.epoch - 1):
            print(f'######## End ###########')
            cleanup()
    torch.cuda.empty_cache()
    if dist.is_initialized(): # 检查进程组是否已被初始化
        dist.barrier()


if __name__ == '__main__':
    # start 
    
    total_ids= [i.split('.')[0] for i in os.listdir(args.graph_file_dir)]

    training_pdb_ids, validation_pdb_ids = train_test_split(total_ids, train_size=0.9, shuffle=True, random_state=42) # random_state=args.random_seed)
    training('cuda:0',
                training_pdb_ids[:], 
                validation_pdb_ids[:], 
                0, 
                0, 
                args)
