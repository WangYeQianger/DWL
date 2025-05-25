#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @author : Xujun Zhang, Tianyue Wang

'''
@File    :   generate_graph.py
@Time    :   2023/02/24 14:21:01
@Author  :   Xujun Zhang, Tianyue Wang
@Version :   1.0
@Contact :   
@License :   
@Desc    :   None
'''

# here put the import lib
import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from dataset import graph_obj

argparser = argparse.ArgumentParser()
argparser.add_argument('--pocket_path', type=str, default='data/CASP/CASP15/pocket')
argparser.add_argument('--graph_file_dir', type=str, default='data/CASP/CASP15/graph')
# argparser.add_argument('--pocket_path', type=str, default='/root/loop/CASP1314_pocket')
# argparser.add_argument('--graph_file_dir', type=str, default='/root/loop/CASP1314_graph')
args = argparser.parse_args()
# init
pocket_path = args.pocket_path
graph_file_dir = args.graph_file_dir
os.makedirs(graph_file_dir, exist_ok=True)
pdb_ids = [pdb.split('_pocket')[0] for pdb in os.listdir(pocket_path) if not os.path.exists(f'{graph_file_dir}/{pdb.split("_pocket")[0]}.dgl')]
# bad_pdb_ids = ['6QUH_A_142_147', '7OIN_A_76_84', '7OIN_A_129_140', 
#                '6QUH_A_49_56', '6QUH_A_209_216', '4OUO_A_187_190', 
#                '6QUH_A_172_175', '7OIN_A_52_58', '6QUH_A_37_40', 
#                '6QUH_A_188_198', '7OIN_A_101_104', '7OIN_A_88_91', 
#                '6QUH_A_101_104', '7OIN_A_206_210', '7OIN_A_169_172',
#                '7OIN_A_9_12']
# pdb_ids = [i for i in pdb_ids if i not in bad_pdb_ids]
# pdb_ids = ['T1038_A_95_101']
test_dataset = graph_obj.LoopGraphDataset(src_dir=pocket_path,
                                        dst_dir=graph_file_dir,
                                        pdb_ids=pdb_ids[:],
                                        pki_labels=None,
                                        dataset_type='test',
                                        n_job=1,
                                        on_the_fly=True,
                                        verbose=True)

