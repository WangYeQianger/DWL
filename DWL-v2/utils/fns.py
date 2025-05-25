#!usr/bin/env python3
# -*- coding:utf-8 -*-
# @time : 2022/8/8 14:18
# @author : Xujun Zhang, Tianyue Wang

import torch
import random
import numpy as np
import os
import sys
from joblib import load, dump
from rdkit import Chem
from rdkit import Geometry
import copy
from torch.distributions import Normal
pwd_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(pwd_dir)


def mdn_loss_fn(pi, sigma, mu, y):
    epsilon = 1e-16
    normal = Normal(mu, sigma+epsilon)
    loglik = normal.log_prob(y.expand_as(normal.loc)) + epsilon
    pi = torch.softmax(pi, dim=1)
    loss = -torch.logsumexp(torch.log(pi) + loglik, dim=1)
    return loss

def set_random_seed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def partition_job(data_lis, job_n, total_job=4, strict=False):
    length = len(data_lis)
    step = length // total_job
    if not strict:
        if job_n == total_job - 1:
            return data_lis[job_n * step:]
    return data_lis[job_n * step: (job_n + 1) * step]


def save_graph(dst_file, data):
    dump(data, dst_file)


def load_graph(src_file):
    return load(src_file)


def partition_job(data_lis, job_n, total_job=4, strict=False):
    length = len(data_lis)
    step = length // total_job
    if length % total_job == 0:
        return data_lis[job_n * step: (job_n + 1) * step]
    else:
        if not strict:
            if job_n == total_job - 1:
                return data_lis[job_n * step:]
            else:
                return data_lis[job_n * step: (job_n + 1) * step]
        else:
            step += 1
            if job_n * step <= length-1:
                data_lis += data_lis
                return data_lis[job_n * step: (job_n + 1) * step]
            else:
                return random.sample(data_lis, step)


def read_equibind_split(split_file):
    with open(split_file, 'r') as f:
        lines = f.read().splitlines()
    return lines


def karmaLoop_evaluation(model, dataset_loader, device, pos_r):
    '''
    used for evaluate model
    :param model:
    :param dataset_loader:
    :param device:
    :return:
    '''
    # do not recompute parameters in batch normalization and dropout
    model.eval()
    total_losses = []
    rmsd_losss = []
    mdntrue_losses = []
    # do not save grad
    with torch.no_grad(): 
        # mini batch
        for idx, batch_data in enumerate(dataset_loader):
            # get data
            data = batch_data
            # to device
            data = data.to(device)
            # forward
            rmsd_loss, mdn_loss_true, _ = model(data, pos_r)
            mdntrue_losses.append(mdn_loss_true.view((-1, 1)))
            loss = pos_r*rmsd_loss + 1*mdn_loss_true
            total_losses.append(loss.view((-1, 1)))
            rmsd_losss.append(rmsd_loss.view((-1, 1)))
        return torch.cat(total_losses), torch.cat(rmsd_losss),torch.cat(mdntrue_losses)


class Early_stopper(object):
    def __init__(self, model_file, mode='higher', patience=70, tolerance=0.0):
        self.model_file = model_file
        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower
        self.patience = patience
        self.tolerance = tolerance
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        # return (score > prev_best_score)
        return score / prev_best_score > 1 + self.tolerance

    def _check_lower(self, score, prev_best_score):
        # return (score < prev_best_score)
        return prev_best_score / score > 1 + self.tolerance

    def load_model(self, model_obj, my_device, strict=False):
        '''Load model saved with early stopping.'''
        model_obj.load_state_dict(torch.load(self.model_file, map_location=my_device)['model_state_dict'], strict=strict)

    def save_model(self, model_obj):
        '''Saves model when the metric on the validation set gets improved.'''
        torch.save({'model_state_dict': model_obj.state_dict()}, self.model_file)

    def step(self, score, model_obj):
        if self.best_score is None:
            self.best_score = score
            self.save_model(model_obj)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_model(model_obj)
            self.counter = 0
        else:
            self.counter += 1
            print(
                f'# EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        print(f'# Current best performance {float(self.best_score):.3f}')
        return self.early_stop
    
def set_loop_positions_rdkit(rdkit_mol, loop_positions, loop_idx_in_mol):
    rdkit_mol_ = copy.deepcopy(rdkit_mol)
    rd_conf = rdkit_mol_.GetConformer()
    for j in range(loop_positions.shape[0]):
        rd_conf.SetAtomPosition(loop_idx_in_mol[j], Geometry.Point3D(*loop_positions[j]))
    return rdkit_mol_

def set_loop_positions(mda_mol, loop_positions, chain, res_num_src, res_num_dst):
    loop_mol = mda_mol.select_atoms(f'chainid {chain} and (resid {res_num_src}:{res_num_dst})')
    # set loop position
    loop_mol.positions = loop_positions
    # return 
    return mda_mol

def make_movide(mol, pos_seq, movie_file, loop_idx_in_mol):
    # pos_seq: numpy.array  shape = (25, atom_num, 3)
    with Chem.SDWriter(movie_file) as w:
        for i in range(pos_seq.shape[0]):
            pos_i = pos_seq[i]
            mol_i = copy.deepcopy(mol)
            mol_i = set_loop_positions_rdkit(mol_i, pos_i, loop_idx_in_mol)
            w.write(mol_i)

def save_loop_file(data, out_dir, out_init=False, out_movie=False, out_pred=True):
    # pocket_centers = data.pocket_center.cpu().numpy().astype(np.float64)
    for idx, mol in enumerate(data.mol):
        loop_idx_in_mol = data.loop_idx_2_mol_idx[idx]
        if out_init:
            # random position
            pos_init = data['loop'].pos[data['loop'].batch==idx].cpu().numpy().astype(np.float64) # + pocket_centers[idx]
            random_mol = set_loop_positions_rdkit(mol, pos_init, loop_idx_in_mol)
            random_file = f'{out_dir}/{data.pdb_id[idx]}_random_pos.pdb'
            Chem.MolToPDBFile(random_mol, random_file)
        if out_movie:
            # make movie
            pos_seq = data.pos_seq[:, data['loop'].batch==idx, :].cpu().numpy().astype(np.float64)
            # movie_file = f'{out_dir}/{data.pdb_id[idx]}/{data.pdb_id[idx]}_pred_movie.sdf'
            movie_file = f'{out_dir}/{data.pdb_id[idx]}_pred_movie.sdf'
            make_movide(mol, pos_seq, movie_file)
        # correct pos
        pos_pred = data.pos_preds[data['loop'].batch==idx].cpu().numpy().astype(np.float64) # + pocket_centers[idx]
        pred_mol = set_loop_positions_rdkit(mol, pos_pred, loop_idx_in_mol)
        if out_pred:
            pred_file = f'{out_dir}/{data.pdb_id[idx]}_pred.pdb'
            Chem.MolToPDBFile(pred_mol, pred_file)
