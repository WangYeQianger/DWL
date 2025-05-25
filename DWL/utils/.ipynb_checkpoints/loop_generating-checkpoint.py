#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @author : Xujun Zhang, Tianyue Wang

'''
@File    :   loop_generating.py
@Time    :   2023/03/09 19:31:32
@Author  :   Xujun Zhang, Tianyue Wang
@Version :   1.0
@Contact :   
@License :   
@Desc    :   None
'''
# here put the import lib
import argparse
import os
import sys
import time
import torch.nn as nn
import pandas as pd
import torch.optim
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm
# dir of current
pwd_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(pwd_dir))
from utils.fns import Early_stopper, set_random_seed, save_loop_file
from dataset.graph_obj import LoopGraphDataset
from dataset.dataloader_obj import PassNoneDataLoader
from architecture.Net_architecture import KarmaLoop

class DataLoaderX(PassNoneDataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

# get parameters from command line
argparser = argparse.ArgumentParser()
argparser.add_argument('--graph_file_dir', type=str,
                        # default='/root/pdb_ab_graph',
                        default='graph_100',
                       help='the graph files path')
argparser.add_argument('--model_file', type=str,
                       default='model_pkls/autoloop.pkl',
                       help='pretrained model file')
argparser.add_argument('--out_dir', type=str,
                       default='loop_result',
                       help='dir for recording loop conformations and scores')
argparser.add_argument('--scoring', type=bool,
                       default=True,
                       help='whether predict loop generating scores')
argparser.add_argument('--save_file', type=bool,
                       default=True,
                       help='whether save predicted loop conformations')
argparser.add_argument('--batch_size', type=int,
                       default=8,
                       help='batch size')
argparser.add_argument('--random_seed', type=int,
                       default=2020,
                       help='random_seed')
args = argparser.parse_args()
set_random_seed(args.random_seed)

# get pdb_ids
test_pdb_ids = [i.split('.')[0] for i in os.listdir(args.graph_file_dir)]
test_pdb_ids = sorted(test_pdb_ids, key=lambda x: int(x.split('_')[-1]) - int(x.split('_')[-2]))
# test_pdb_ids = ['T1024_A_185_194']
# dataset
test_dataset = LoopGraphDataset(src_dir='',
                                dst_dir=args.graph_file_dir,
                                pdb_ids=test_pdb_ids[:],
                                dataset_type='test',
                                random_forward_reverse=False,
                                n_job=1, 
                                on_the_fly=True)

# dataloader
test_dataloader = DataLoaderX(dataset=test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)
# device
device_id = 0
if torch.cuda.is_available():
    my_device = f'cuda:{device_id}'
else:
    my_device = 'cpu'
# model
model = KarmaLoop(hierarchical=True)
model = nn.DataParallel(model, device_ids=[device_id], output_device=device_id)
model.to(my_device)
# stoper
stopper = Early_stopper(model_file=args.model_file,
                        mode='lower', patience=10)
# model_name = args.model_file.split('/')[-1].split('_')[1]
print('# load model')
# load existing model
stopper.load_model(model_obj=model, my_device=my_device, strict=True)
# time
start_time = time.perf_counter()
total_time = 0
data_statistic = []
time_info=[]
for re in range(1):
    rmsds = torch.as_tensor([]).to(my_device)
    confidence_scores = []
    pdb_ids = []
    ff_corrected_rmsds = []
    align_corrected_rmsds = []
    model.eval()
    egnn_time = 0
    out_dir_r = f'{args.out_dir}/{re}'
    os.makedirs(out_dir_r, exist_ok=True)
    with torch.no_grad():
        for idx, data in enumerate(tqdm(test_dataloader)):
            # to device
            data = data.to(my_device)
            start_ = time.time()
            # forward
            egnn_start_time = time.perf_counter()
            rmsd_loss, mdn_score, pos_pred = model.module.forward(data, scoring=args.scoring, dist_threhold=5, train=False)
            # print(rmsd_loss)
            egnn_time += time.perf_counter() - egnn_start_time
            pos_true = data['loop'].xyz
            batch = data['loop'].batch
            # print(torch.unique(data['loop'].batch, return_counts=True)[1])
            end_=time.time()
            per_time = end_-start_
            time_info.append([data.pdb_id, per_time])
            pos_loss = model.module.cal_rmsd(pos_true, pos_pred, batch) 
            rmsds = torch.cat([rmsds, pos_loss], dim=0)
            # output conformation
            if args.save_file:
                data.pos_preds = pos_pred
                save_loop_file(data, out_dir=out_dir_r, out_init=False, out_movie=False, out_pred=True)
            pdb_ids.extend(data.pdb_id)
            confidence_scores.extend(mdn_score.cpu().numpy().tolist())
        if args.scoring:
            df = pd.DataFrame({'pdb_id':pdb_ids, 'confidence score': confidence_scores})
            df.to_csv(f'{args.out_dir}/score.csv', index=False)
        # report 
        data_statistic.append([rmsds.mean(), 
                               rmsds.median(), 
                               rmsds.max(), 
                               rmsds.min(), 
                               (rmsds<=5).sum()/rmsds.size(0), 
                               (rmsds<=4.5).sum()/rmsds.size(0), 
                               (rmsds<=4).sum()/rmsds.size(0), 
                               (rmsds<=3.5).sum()/rmsds.size(0), 
                               (rmsds<=3).sum()/rmsds.size(0), 
                               (rmsds<=2.5).sum()/rmsds.size(0), 
                               (rmsds<=2).sum()/rmsds.size(0), 
                               (rmsds<=1).sum()/rmsds.size(0), 
                               egnn_time / 60, 
                               ])
data_statistic_mean = torch.as_tensor(data_statistic).mean(dim=0)
data_statistic_std = torch.as_tensor(data_statistic).std(dim=0)
prediction_time = time.perf_counter()
print(f'''
Total Time: {(prediction_time - start_time) / 60} min
Sample Num: {len(test_dataset)}
# prediction
Time Spend: {data_statistic_mean[12]} ± {data_statistic_std[12]} min
Mean RMSD: {data_statistic_mean[0]} ± {data_statistic_std[0]}   
Medium RMSD: {data_statistic_mean[1]} ± {data_statistic_std[1]}
Max RMSD: {data_statistic_mean[2]} ± {data_statistic_std[2]}
Min RMSD: {data_statistic_mean[3]} ± {data_statistic_std[3]}
Success RATE(5A): {data_statistic_mean[4]} ± {data_statistic_std[4]}
Success RATE(4.5A): {data_statistic_mean[5]} ± {data_statistic_std[5]}
Success RATE(4A): {data_statistic_mean[6]} ± {data_statistic_std[6]}
Success RATE(3.5A): {data_statistic_mean[7]} ± {data_statistic_std[7]}
Success RATE(3A): {data_statistic_mean[8]} ± {data_statistic_std[8]}
Success RATE(2.5A): {data_statistic_mean[9]} ± {data_statistic_std[9]}
Success RATE(2A): {data_statistic_mean[10]} ± {data_statistic_std[10]}
Success RATE(1A): {data_statistic_mean[11]} ± {data_statistic_std[11]}''')

