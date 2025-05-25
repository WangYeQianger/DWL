#!usr/bin/env python3
# -*- coding:utf-8 -*-
# @time : 2022/3/28 14:08
# @author : Xujun Zhang, Tianyue Wang
import prody
import os
import sys
import MDAnalysis as mda
from functools import partial 
from multiprocessing import Pool
import numpy as np
import torch
import multiprocessing
from rdkit import Chem
from rdkit import RDLogger
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
RDLogger.DisableLog("rdApp.*")
# dir of current
from utils.fns import load_graph, save_graph
from dataset.protein_feature import get_protein_feature_mda
from dataset.loop_feature import get_loop_feature_strict, get_loop_feature_strict_rot


print = partial(print, flush=True)
   
class LoopGraphDataset(Dataset):

    def __init__(self, src_dir, pdb_ids, dst_dir, pki_labels=None, dataset_type='train', n_job=1,
                 on_the_fly=False,
                 random_forward_reverse=True,
                 verbose=False):
        '''
        :param on_the_fly: whether to get graph from a totoal graph list or a single graph file
        _______________________________________________________________________________________________________
        |  mode  |  generate single graph file  |  generate integrated graph file  |  load to memory at once  |
        |  False |          No                  |              Yes                 |            Yes           |
        |  True  |          Yes                 |              No                  |            No            |
        |  Fake  |          Yes                 |              No                  |            Yes           |
        _______________________________________________________________________________________________________
        '''
        self.src_dir = src_dir
        self.pdb_ids = pdb_ids
        self.dst_dir = dst_dir
        if pki_labels is not None:
            self.pki_labels = pki_labels
        else:
            self.pki_labels = np.zeros((len(self.pdb_ids)))
        os.makedirs(dst_dir, exist_ok=True)
        assert dataset_type in ['train', 'valid', 'test'], 'illegal dataset type'
        self.dataset_type = dataset_type
        self.random_forward_reverse = random_forward_reverse
        self.dst_file = f'{dst_dir}/{dataset_type}.dgl'
        self.n_job = n_job
        assert on_the_fly in [True, False, 'Fake']
        self.verbose = verbose
        self.on_the_fly = on_the_fly
        self.graph_labels = []
        self.pre_process()

    def pre_process(self):
        if self.on_the_fly == 'Fake':
            self._generate_graph_on_the_fly_fake()
        elif self.on_the_fly:
            self._generate_graph_on_the_fly()
        else:
            self._generate_graph()

    def _generate_graph(self):
        if os.path.exists(self.dst_file):
            if self.verbose:
                print('load graph')
            self.graph_labels = load_graph(self.dst_file)
        else:
            idxs = range(len(self.pdb_ids))
            if self.verbose:
                print('### cal graph')
            single_process = partial(self._single_process, return_graph=True, save_file=False)
            # generate graph
            if self.n_job == 1:
                if self.verbose:
                    idxs = tqdm(idxs)
                for idx in idxs:
                    self.graph_labels.append(single_process(idx))
            else:
                pool = Pool(self.n_job)
                self.graph_labels = pool.map(single_process, idxs)
                pool.close()
                pool.join()
            # filter None
            self.graph_labels = list(filter(lambda x: x is not None, self.graph_labels))
            # save file
            save_graph(self.dst_file, self.graph_labels)

    def _generate_graph_on_the_fly(self):
        idxs = range(len(self.pdb_ids))
        if self.verbose:
            print('### get graph on the fly')
        single_process = partial(self._single_process, return_graph=True, save_file=True)
        # generate graph
        if self.n_job == 1:
            if self.verbose:
                idxs = tqdm(idxs)
            for idx in idxs:
                single_process(idx)
        else:
            pool = Pool(self.n_job)
            pool.map(single_process, idxs)
            pool.close()
            pool.join()
        # self.pdb_ids = [os.path.split(i)[-1].split('.')[0] for i in glob.glob(f'{self.dst_dir}/*.dgl')]

    def _generate_graph_on_the_fly_fake(self):
        idxs = range(len(self.pdb_ids))
        if self.verbose:
            print('### get graph on the fly (fake)')
        single_process = partial(self._single_process, return_graph=True, save_file=True)
        # generate graph
        if self.n_job == 1:
            if self.verbose:
                idxs = tqdm(idxs)
            for idx in idxs:
                self.graph_labels.append(single_process(idx))
        else:
            pool = Pool(self.n_job)
            self.graph_labels = pool.map(single_process, idxs)
            pool.close()
            pool.join()
        # filter None
        self.graph_labels = list(filter(lambda x: x is not None, self.graph_labels))

    def _single_process(self, idx, return_graph=False, save_file=False):
        pdb_id = self.pdb_ids[idx]
        # print(f'{pdb_id}')
        dst_file = f'{self.dst_dir}/{pdb_id}.dgl'
        # data = get_loop_graph_v1(pdb_id=pdb_id,
        #                 path=self.src_dir)
        if os.path.exists(dst_file):
            # pass
            # reload graph
            if return_graph:
                return load_graph(dst_file)
        else:
            # generate graph
            # try:
            data = get_loop_graph_v1(pdb_id=pdb_id,
                                    path=self.src_dir)
            data.pdb_id = pdb_id
            if save_file:
                save_graph(dst_file, data)
            if return_graph:
                return data
            # except Exception as e:
            #     print(f'{pdb_id} error because {e}')
            #     return None

    def __getitem__(self, idx):
        if self.on_the_fly == True:
            data = self._single_process(idx=idx, return_graph=True, save_file=False)
        else:
            data = self.graph_labels[idx]
        # decide whether to use forward or reverse
        if data.forward_ok and data.reverse_ok:
            data = self.modyfy_graph_to_forward(data)
            if self.random_forward_reverse:
                if np.random.rand() < 0.5:
                    data = self.modyfy_graph_to_reverse(data)
        elif data.forward_ok:
            data = self.modyfy_graph_to_forward(data)
        elif data.reverse_ok:
            data = self.modyfy_graph_to_reverse(data)
        else:
            return None
        # del data['loop'].seq_parent_forward, data['loop'].seq_order_forward, data['loop'].seq_parent_reverse, data['loop'].seq_order_reverse, data.loop_parent_mask_forward, data.loop_parent_mask_reverse
        return data
    
    def modyfy_graph_to_forward(self, data):
            data['loop'].seq_parent = data['loop'].seq_parent_forward
            data['loop'].seq_order = data['loop'].seq_order_forward
            data.loop_parent_mask = data.loop_parent_mask_forward
            data['protein'].xyz = data['protein'].xyz_full[:, 2]
            return data
    
    def modyfy_graph_to_reverse(self, data):
            data['loop'].seq_parent = data['loop'].seq_parent_reverse
            data['loop'].seq_order = data['loop'].seq_order_reverse
            data.loop_parent_mask = data.loop_parent_mask_reverse
            data['protein'].xyz = data['protein'].xyz_full[:, 0]
            return data

    def __len__(self):
        if self.on_the_fly == True:
            return len(self.pdb_ids)
        else:
            return len(self.graph_labels)



def get_repeat_node(src_num, dst_num):
    return torch.arange(src_num, dtype=torch.long).repeat(dst_num), \
           torch.as_tensor(np.repeat(np.arange(dst_num), src_num), dtype=torch.long)

def pdb2rdmol(pocket_pdb):
    pocket_atom_mol = Chem.MolFromPDBFile(pocket_pdb, removeHs=False, sanitize=False)
    pocket_atom_mol = Chem.RemoveAllHs(pocket_atom_mol, sanitize=False)
    return pocket_atom_mol

def get_loop_graph_v1(pdb_id, path):
    torch.set_num_threads(1)
    pocket_pdb = f'{path}/{pdb_id}_pocket_12A.pdb'
    pdbid, chain, res_num_src, res_num_dst = pdb_id.split('_')
    loop_len = int(res_num_dst) - int(res_num_src) + 1
    # get protein mol
    pocket_mol = mda.Universe(pocket_pdb)
    non_loop_mol = pocket_mol.select_atoms(f'chainid {chain} and not (resid {res_num_src}:{res_num_dst})')
    loop_mol = pocket_mol.select_atoms(f'chainid {chain} and (resid {res_num_src}:{res_num_dst})')
    # assert loop_len == len(loop_mol.residues), f'{pdb_id} loop length error'
    loop_res = [f'{res.atoms.chainIDs[0]}-{res.resid}{res.icode}' for res in loop_mol.residues]
    pocket_atom_mol = pdb2rdmol(pocket_pdb)
    # generate graph
    p_xyz, p_xyz_full, p_seq, p_node_s, p_node_v, p_edge_index, p_edge_s, p_edge_v, p_full_edge_s, p_node_name, p_node_type = get_protein_feature_mda(non_loop_mol)
    loop_xyz, pa_node_feature, pa_edge_index, pa_edge_feature, atom2nonloopres, nonloop_mask, loop_edge_index, loop_edge_feature, loop_cov_edge_mask, loop_idx_2_mol_idx, loop_bb_atom_mask, loop_parent_mask_forward, loop_parent_mask_reverse, seq_order_forward, seq_parent_forward, seq_order_reverse, seq_parent_reverse, forward_ok, reverse_ok = get_loop_feature_strict(pocket_atom_mol, nonloop_res=p_node_name, loop_res=loop_res)
    assert len(p_node_s) == len(loop_parent_mask_forward), f'{pdb_id} loop parent mask error'
    assert len(p_node_s) == len(loop_parent_mask_reverse), f'{pdb_id} loop parent mask error'
    # to data
    data = HeteroData()
    # protein residue
    data.forward_ok = forward_ok
    data.reverse_ok = reverse_ok
    data.atom2nonloopres = torch.tensor(atom2nonloopres, dtype=torch.long) # 
    data.loop_parent_mask_forward = torch.tensor(loop_parent_mask_forward, dtype=torch.bool) # 
    data.loop_parent_mask_reverse = torch.tensor(loop_parent_mask_reverse, dtype=torch.bool) # 
    data.nonloop_mask = torch.tensor(nonloop_mask, dtype=torch.bool) # 
    data.loop_bb_atom_mask = torch.tensor(loop_bb_atom_mask, dtype=torch.bool) # 
    data.loop_cov_edge_mask = loop_cov_edge_mask  # 
    data.loop_idx_2_mol_idx = loop_idx_2_mol_idx # 
    data.mol = pocket_atom_mol
    # data['protein'].node_name = p_node_type
    data['protein'].node_s = p_node_s.to(torch.float32) 
    data['protein'].node_v = p_node_v.to(torch.float32)
    data['protein'].xyz = p_xyz.to(torch.float32) 
    data['protein'].xyz_full = p_xyz_full.to(torch.float32) 
    data['protein'].seq = p_seq.to(torch.int32)
    data['protein', 'p2p', 'protein'].edge_index = p_edge_index.to(torch.long)
    data['protein', 'p2p', 'protein'].edge_s = p_edge_s.to(torch.float32) 
    data['protein', 'p2p', 'protein'].full_edge_s = p_full_edge_s.to(torch.float32) 
    data['protein', 'p2p', 'protein'].edge_v = p_edge_v.to(torch.float32) 
    # protein atom
    data['protein_atom'].node_s = pa_node_feature.to(torch.float32) 
    data['protein_atom', 'pa2pa', 'protein_atom'].edge_index = pa_edge_index.to(torch.long)
    data['protein_atom', 'pa2pa', 'protein_atom'].edge_s = pa_edge_feature.to(torch.float32) 
    # # loop
    # data['loop'].node_name = loop_name
    data['loop'].node_s = torch.zeros((data.loop_bb_atom_mask.sum(), 1))
    data['loop'].xyz = loop_xyz.to(torch.float32)
    data['loop'].seq_parent_forward = seq_parent_forward.to(torch.long)
    data['loop'].seq_order_forward = seq_order_forward.to(torch.long)
    data['loop'].seq_parent_reverse = seq_parent_reverse.to(torch.long)
    data['loop'].seq_order_reverse = seq_order_reverse.to(torch.long)
    data['loop', 'l2l', 'loop'].edge_index = loop_edge_index.to(torch.long)
    data['loop', 'l2l', 'loop'].full_edge_s = loop_edge_feature.to(torch.float32)
    # # protein-loop
    data['protein', 'p2l', 'loop'].edge_index = torch.stack(
        get_repeat_node(p_xyz.shape[0], loop_xyz.shape[0]), dim=0)
    return data



if __name__ == '__main__':
    import pandas as pd
    from tqdm import tqdm
    path = '/root/loop/CASP1314_pocket'
    # df = pd.read_csv('/root/bad_pdb_ids.csv')
    # df = df[df['error_type'] == 0]
    # pdb_ids = df.pdb_id.values # 
    pdb_ids = ['T1038_A_95_101']  # T1037_A_30_34
    for idx, pdb_id in enumerate(tqdm(pdb_ids)):
        dst_file = f'/root/pdb_ab_graph/{pdb_id}.dgl'
        try:
            data = get_loop_graph_v1(pdb_id, path)
            data.pdb_id = pdb_id
            # save_graph(dst_file, data)
        except Exception as e:
            print(idx, e)
            continue
            