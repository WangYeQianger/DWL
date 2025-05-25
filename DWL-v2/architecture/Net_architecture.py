#!usr/bin/env python3
# -*- coding:utf-8 -*-
# @time : 2022/8/8 16:51
# @author : Xujun Zhang, Tianyue Wang

import torch
from torch import nn
import torch.nn.functional as F
from architecture.GVP_Block import GVP_embedding
from architecture.GraphTransformer_Block import GraghTransformer
from architecture.MDN_Block import MDN_Block
from architecture.ImprovedMDN_Block import ImprovedMDN_Block
from architecture.EGNN_Block import EGNN
from architecture.Gate_Block import Gate_Block
from torch_scatter import scatter_mean, scatter
from torch_geometric.nn import GraphNorm
from torch_geometric.utils import to_dense_batch
from utils.fns import mdn_loss_fn


class KarmaLoop(nn.Module):
    def __init__(self, hierarchical=True):
        super(KarmaLoop, self).__init__()
        self.hierarchical = hierarchical
        # encoders
        self.loop_encoder = GraghTransformer(
            in_channels=89, 
            edge_features=20, 
            num_hidden_channels=128,
            activ_fn=torch.nn.SiLU(),
            transformer_residual=True,
            num_attention_heads=4,
            norm_to_apply='batch',
            dropout_rate=0.15,
            num_layers=6
        )
        self.pro_encoder = GVP_embedding(
            (95, 3), (128, 16), (85, 1), (32, 1), seq_in=True, vector_gate=True) 
        self.gn = GraphNorm(128)
        # pose prediction
        self.pose_sampling_layers = nn.ModuleList([EGNN(dim_in=128, dim_tmp=128, edge_in=128, edge_out=128, num_head=4, drop_rate=0.15, update_pos=False) if i < 7 else EGNN(dim_in=128, dim_tmp=128, edge_in=128, edge_out=128, num_head=4, drop_rate=0.15, update_pos=True) for i in range(8) ])
        self.edge_init_layer = nn.Linear(6, 128)
        if hierarchical:
            self.merge_hierarchical = nn.Sequential(
            nn.Linear(256, 256),
            nn.Dropout(p=0.15),
            nn.LeakyReLU(),
            nn.Linear(256, 128)
        )
            self.attn_fusion = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
            self.alpha = nn.Parameter(torch.tensor(1.0))
            self.graph_norm = GraphNorm(in_channels=128)
        # scoring 
        self.mdn_layer = MDN_Block(hidden_dim=128, 
                                         n_gaussians=10, 
                                        dropout_rate=0.10, 
                                        dist_threhold=7.)


    def cal_rmsd(self, pos_ture, pos_pred, batch, if_r=True):
        if if_r:
            return scatter_mean(((pos_pred - pos_ture)**2).sum(dim=-1), batch).sqrt()
        else:
            return scatter_mean(((pos_pred - pos_ture)**2).sum(dim=-1), batch)
    
    def forward(self, data, pos_r=1, scoring=True, dist_threhold=5, train=True):
    
        '''
        generating loop conformations and  predicting the confidence score
        '''
        # print(data.pdb_id)
        device = data['protein'].node_s.device
        batch_size = data['protein'].batch.max()+1
        # encoder
        pro_node_s, loop_node_s, data, pro_nodes_num = self.encoding(data)
        if pos_r:
            # interaction graph
            data, xyz, batch_seq_order, batch_seq_parent, generate_mask_from_protein, generate_mask_from_lig, batch = self.construct_interaction_graph(data, pro_nodes_num, train=train)
            # loop_modelling
            loop_pos, pred_xyz, true_xyz, sample_batch = self.sampling_atom_pos(data, xyz, batch_seq_order, batch_seq_parent, generate_mask_from_protein, generate_mask_from_lig, pro_nodes_num, batch, train=train)
            loop_pos = loop_pos[pro_nodes_num:]
            # loss
            rmsd_losss = self.cal_rmsd(pos_ture=true_xyz, pos_pred=pred_xyz, batch=sample_batch, if_r=True)
        else:
            loop_pos = data['loop'].xyz
            rmsd_losss = torch.zeros(1, device=device, dtype=torch.float)
        # scoring
        if pos_r:
            # 使用预测坐标
            scoring_loop_pos = loop_pos  # 这是预测的坐标
        else:
            # 如果没有预测，可能需要初始化或使用其他策略
            scoring_loop_pos = data['loop'].xyz  # 或者某种初始化坐标
        # if scoring:
        #     mdn_score = self.scoring(loop_s=loop_node_s, loop_pos=data['loop'].xyz, pro_s=pro_node_s, data=data,
        #                                                        dist_threhold=dist_threhold, batch_size=batch_size, train=train)
        if scoring:
            mdn_score = self.scoring(loop_s=loop_node_s, loop_pos=scoring_loop_pos, pro_s=pro_node_s, data=data,
                                                               dist_threhold=dist_threhold, batch_size=batch_size, train=train)
        else:
            mdn_score = torch.zeros(1, device=device, dtype=torch.float)
        if train:     
            rmsd_losss = rmsd_losss.mean()
            mdn_score = mdn_score.mean()
        return rmsd_losss, mdn_score, loop_pos
        
    def encoding(self, data):
        '''
        get loop & protein embeddings
        '''
        # encoder 
        proatom_node_s = self.loop_encoder(data['protein_atom'].node_s.to(torch.float32), data['protein_atom', 'pa2pa', 'protein_atom'].edge_s.to(torch.float32), data['protein_atom', 'pa2pa', 'protein_atom'].edge_index)
        pro_node_s = self.pro_encoder((data['protein']['node_s'], data['protein']['node_v']),
                                                      data[(
                                                          "protein", "p2p", "protein")]["edge_index"],
                                                      (data[("protein", "p2p", "protein")]["edge_s"],
                                                       data[("protein", "p2p", "protein")]["edge_v"]),
                                                      data['protein'].seq)
        loop_node_s = proatom_node_s[data.loop_bb_atom_mask]
        proatom_node_s = proatom_node_s[~data.nonloop_mask]
        proatom_node_s = scatter(proatom_node_s, index=data.atom2nonloopres, reduce='sum', dim=0)
        proatom_node_s = self.graph_norm(proatom_node_s, data['protein'].batch)
        # pro_node_s = torch.cat([pro_node_s, proatom_node_s], dim=-1)
        # pro_node_s = self.merge_hierarchical(pro_node_s)
        concat_feat = torch.cat([pro_node_s, proatom_node_s], dim=-1)
        merged_feat = self.merge_hierarchical(concat_feat)  # [N, 128]

        # 使用 self-attention 将 proatom_node_s 融合进 pro_node_s
        # 注意力输入要求 3D: [B, N, D]，你这里是 batchless图，处理方式如下：
        query = pro_node_s.unsqueeze(0)     # [1, N, 128]
        key = proatom_node_s.unsqueeze(0)   # [1, N, 128]
        value = proatom_node_s.unsqueeze(0) # [1, N, 128]

        attn_out, _ = self.attn_fusion(query, key, value)  # [1, N, 128]
        attention_feat = attn_out.squeeze(0)               # [N, 128]

        # 加权组合
        alpha = self.alpha
        pro_node_s = alpha * merged_feat + (1 - alpha) * attention_feat
        
        # graph norm through a protein-loop_bb graph
        pro_nodes_num = data['protein'].num_nodes
        node_s = self.gn(torch.cat([pro_node_s, loop_node_s], dim=0), torch.cat([data['protein'].batch, data['loop'].batch], dim=-1))
        data['protein'].node_s, data['loop'].node_s = node_s[:pro_nodes_num], node_s[pro_nodes_num:]
        return pro_node_s, loop_node_s, data, pro_nodes_num
    
    def scoring(self, loop_s, loop_pos, pro_s, data, dist_threhold, batch_size, train=True):
        '''
        confidence score
        '''
        pi, sigma, mu, dist, c_batch, atom_types, bond_types = self.mdn_layer(loop_s=loop_s, loop_pos=loop_pos, loop_batch=data['loop'].batch,
                                                               pro_s=pro_s, pro_pos=data['protein'].xyz_full, pro_batch=data['protein'].batch,
                                                               edge_index=data['loop', 'l2l', 'loop'].edge_index[:, data.loop_cov_edge_mask])
        # mdn aux labels
        if train:
            aux_r = 0.001
            # atom_types_label = torch.argmax(data['loop'].node_s[:,:18], dim=1, keepdim=False)
            # bond_types_label = torch.argmax(data['loop', 'l2l', 'loop'].edge_s[data['loop'].cov_edge_mask][:, :5], dim=1, keepdim=False)
            mdn_score = mdn_loss_fn(pi, sigma, mu, dist)[torch.where(dist <= self.mdn_layer.dist_threhold)[0]].mean().float() # + aux_r*F.cross_entropy(atom_types, atom_types_label) # + aux_r*F.cross_entropy(bond_types, bond_types_label) 
        else:
            mdn_score = self.mdn_layer.calculate_probablity(pi, sigma, mu, dist)
            mdn_score[torch.where(dist > dist_threhold)[0]] = 0.
            mdn_score = scatter(mdn_score, index=c_batch, dim=0, reduce='sum', dim_size=batch_size).float()
        return mdn_score
    
    def construct_interaction_graph(self, data, pro_nodes_num, train=True):
        batch_seq_order, _ = to_dense_batch(data['loop'].seq_order, batch=data['loop'].batch, fill_value=9999)
        batch_seq_parent, _ = to_dense_batch(data['loop'].seq_parent, batch=data['loop'].batch, fill_value=9999)
        # transform lig batch idx to pytorch geometric batch idx
        lig_nodes_num_per_sample = torch.unique(data['loop'].batch, return_counts=True)[1]
        lig_nodes_num_per_sample = torch.cumsum(lig_nodes_num_per_sample, dim=0)
        batch_seq_order += pro_nodes_num
        batch_seq_order[1:] += lig_nodes_num_per_sample[:-1].view((-1, 1)) 
        batch_seq_parent += torch.cat([torch.zeros((1,1)).to(lig_nodes_num_per_sample.device), lig_nodes_num_per_sample[:-1].view((-1, 1))], dim=0).to(torch.long) + pro_nodes_num
        # get parent idx
        batch_seq_parent[:, 0] = torch.arange(0, data.loop_parent_mask.size(0)).to(data['loop'].node_s.device)[data.loop_parent_mask]
        # construct interaction graph
        batch = torch.cat([data['protein'].batch, data['loop'].batch], dim=-1)
        # edge index
        u = torch.cat([
            data[("protein", "p2p", "protein")]["edge_index"][0], 
            data[('loop', 'l2l', 'loop')]["edge_index"][0]+pro_nodes_num, 
            data[('protein', 'p2l', 'loop')]["edge_index"][0], data[('protein', 'p2l', 'loop')]["edge_index"][1]+pro_nodes_num], dim=-1)
        v = torch.cat([
            data[("protein", "p2p", "protein")]["edge_index"][1], 
            data[('loop', 'l2l', 'loop')]["edge_index"][1]+pro_nodes_num, 
            data[('protein', 'p2l', 'loop')]["edge_index"][1]+pro_nodes_num, data[('protein', 'p2l', 'loop')]["edge_index"][0]], dim=-1)
        edge_index = torch.stack([u, v], dim=0)
        # pose
        xyz = torch.cat([data['protein'].xyz, data['loop'].xyz], dim=0)   
        dist = torch.pairwise_distance(xyz[edge_index[0]], xyz[edge_index[1]], keepdim=True)
        # features
        node_s = torch.cat([data['protein'].node_s, data['loop'].node_s], dim=0)
        # node_v = torch.cat([data['protein'].xyz, torch.zeros_like(data['ligand'].xyz)], dim=0).unsqueeze(dim=1)
        edge_s = torch.zeros((data[('protein', 'p2l', 'loop')]["edge_index"][0].size(0)*2, 6), device=node_s.device)
        edge_s[:, -1] = -1
        edge_s = torch.cat([data[("protein", "p2p", "protein")].full_edge_s, data['loop', 'l2l', 'loop'].full_edge_s, edge_s], dim=0)
        # edge_v = torch.cat([node_v[edge_index[0]] - node_v[edge_index[1]], node_v[edge_index[1]] - node_v[edge_index[0]]], dim=1)
        edge_s = self.edge_init_layer(edge_s)
        # generation mask
        generate_mask_from_protein = torch.ones((node_s.size(0), 1), device=node_s.device)
        generate_mask_from_protein[pro_nodes_num:] = 0
        # generate_mask_from_protein[batch_seq_parent[:, 0]] = 1
        # for random masking in training process
        generate_mask_from_lig = torch.ones((node_s.size(0), 1), device=node_s.device)
        if train:
            # random choose six consecutive nodes for model training
            num_consecutive_nodes = 10
            # get node number for each sample in a batch. 
            num_nodes_per_batch = torch.unique(data['loop'].batch, return_counts=True)[1]
            # random select start index for each sample in a batch; start from 0
            start_indices = torch.clamp(num_nodes_per_batch - num_consecutive_nodes, min=0)
            start_indices = torch.clamp(torch.rand_like(start_indices.float()) * start_indices, max=batch_seq_order.size(1)-1-num_consecutive_nodes).long() # + 1
            # random set 1/3 samples to zero
            start_indices[torch.randperm(start_indices.size(0))[:start_indices.size(0)//3]] = 0
            # get end index for each sample in a batch
            end_indices = start_indices + num_consecutive_nodes
            # get batch select index
            batch_select_index = torch.arange(batch_seq_order.size(1)).unsqueeze(0).to(start_indices.device)
            # get six consecutive nodes index for each sample in a batch
            predict_batch_select_index = (batch_select_index >= start_indices.unsqueeze(1)) & (batch_select_index < end_indices.unsqueeze(1))
            # get nodes with masked coordinates for each sample in a batch
            start_batch_select_index = (batch_select_index >= start_indices.unsqueeze(1)) 
            start_idx = batch_seq_order[start_batch_select_index]
            start_idx = start_idx[start_idx < pro_nodes_num + 9999]
            # mask nodes
            generate_mask_from_lig[start_idx] = 0
            batch_seq_order = batch_seq_order[predict_batch_select_index].view((-1, num_consecutive_nodes))
            batch_seq_parent = batch_seq_parent[predict_batch_select_index].view((-1, num_consecutive_nodes))
            # # pos[start_idx] = torch.zeros_like(pos[start_idx])
            # # update batch_seq_order and batch_seq_parent
            # batch_seq_order = torch.cat([
            #     batch_seq_order[:, :1],
            #     batch_seq_order[predict_batch_select_index].view((-1, num_consecutive_nodes))]
            #                             , dim=1)
            # batch_seq_parent = torch.cat([
            #     batch_seq_parent[:, :1],
            #     batch_seq_parent[predict_batch_select_index].view((-1, num_consecutive_nodes))
            # ], dim=1)
            # # pos[start_idx:] = torch.zeros_like(pos[start_idx:])
        # to data
        data.edge_index, data.edge_s, data.node_s, data.dist = edge_index, edge_s, node_s, dist
        # data.edge_v, data.node_v = edge_v, node_v
        return data, xyz, batch_seq_order, batch_seq_parent, generate_mask_from_protein, generate_mask_from_lig, batch
    
    def sampling_atom_pos(self, data, xyz, batch_seq_order, batch_seq_parent, generate_mask_from_protein, generate_mask_from_lig, pro_nodes_num, batch, train=True):
        pred_xyz = []
        true_xyz = []
        sample_batch = []
        # pos = xyz.clone()
        xyz = xyz.clone().detach()
        pos = xyz.clone().detach()
        # add noise to xyz for generation stability
        # if train:
        #     pos[pro_nodes_num:] = torch.randn_like(pos[pro_nodes_num:])*0.7 + pos[pro_nodes_num:]
        # c_pos = xyz.clone()
        generate_mask = generate_mask_from_lig if train else generate_mask_from_protein
        # print(batch_seq_order.size(1))
        for r_idx in range(batch_seq_order.size(1)):
            # get node_s, edge_index, edge_s
            node_s, edge_index, edge_s, = data.node_s, data.edge_index, data.edge_s
            # node_v, edge_v  = data.node_v, data.edge_v
            # raw_node_v = node_v.clone()
            # get parent node idxes and generate node idxes
            parent_node_idxes = batch_seq_parent[:, r_idx]
            generate_node_idxes = batch_seq_order[:, r_idx]
            parent_node_idxes = parent_node_idxes[parent_node_idxes < pro_nodes_num + 9999]
            generate_node_idxes = generate_node_idxes[generate_node_idxes < pro_nodes_num + 9999]
            generate_node_dist = torch.pairwise_distance(xyz[parent_node_idxes], xyz[generate_node_idxes], keepdim=True)
            if (generate_node_dist > 2).any():
                print(generate_node_dist[generate_node_dist > 2])
            ### mask unkonwn distance
            generate_mask[generate_node_idxes] = 1
            mask_inv = (generate_mask == 0)
            mask_edge_inv = mask_inv[edge_index[0]] | mask_inv[edge_index[1]] 
            # init next node pos
            pos_ = pos.clone()
            # # give protein node a C atom coordinate
            # protein_catom_mask = parent_node_idxes < pro_nodes_num 
            # pos_[parent_node_idxes][protein_catom_mask] = data.c_xyz[parent_node_idxes][protein_catom_mask]
            # init pos
            pos_[generate_node_idxes] = pos_[parent_node_idxes] + torch.clamp(torch.randn_like(pos_[parent_node_idxes])*0.01, min=-0.01, max=0.01)
            # init vector feats
            # node_v = self.node_v_init_layer(node_v)
            # edge_v = self.edge_v_init_layer(edge_v)
            # egnn layers for pose prediction
            for idx, layer in enumerate(self.pose_sampling_layers):
                # node_s, node_v, edge_s, edge_v, pred_pose = layer(node_s, node_v, edge_s, edge_v, edge_index, generate_node_dist, xyz, parent_node_idxes, generate_node_idxes, mask_edge_inv, pro_nodes_num)
                node_s, edge_s, delta_x = layer(node_s, edge_s, edge_index, generate_node_dist, pos_, parent_node_idxes, generate_node_idxes, mask_edge_inv, pro_nodes_num, batch)
            # # res-connection during each recycling
            # generate_mask[generate_node_idxes] = 1
            # for training
            sample_batch.append(data['loop'].batch[generate_node_idxes-pro_nodes_num])
            true_xyz.append(xyz[generate_node_idxes, :] - xyz[parent_node_idxes, :])
            pred_xyz.append(delta_x)
            # update vector feats
            if train:
                update_pos_mask = torch.rand_like(generate_node_idxes.float()) > 0.5
                generate_node_idxes = generate_node_idxes[update_pos_mask]
                delta_x = delta_x[update_pos_mask].clone().detach()
                # if r_idx == 0:
                #     print(f'{r_idx}: {self.cal_rmsd(pred_pose, xyz[generate_node_idxes], batch=data["loop"].batch[generate_node_idxes-pro_nodes_num]).mean()}')
                # update pos
            pos[generate_node_idxes] =  pos_[generate_node_idxes].clone().detach() + delta_x
                # c_pos[generate_node_idxes] = pred_pose
            # raw_node_v[generate_node_idxes] = pred_pose.unsqueeze(1)
            # data.node_v = raw_node_v
            # data.edge_v = torch.cat([data.node_v[edge_index[0]] - data.node_v[edge_index[1]], data.node_v[edge_index[1]] - data.node_v[edge_index[0]]], dim=1)
        return pos, torch.cat(pred_xyz, dim=0), torch.cat(true_xyz, dim=0), torch.cat(sample_batch, dim=0)
    
