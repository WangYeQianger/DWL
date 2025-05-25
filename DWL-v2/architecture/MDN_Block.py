#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   MDN Block
@Time    :   2022/09/10 10:34:28
@Author  :   copied from DeepDock
@Version :   1.0
@Contact :   
@License :   
@Desc    :   None
'''

# here put the import lib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch_geometric.utils import to_dense_batch




class MDN_Block(nn.Module):
    def __init__(self, hidden_dim, n_gaussians, dropout_rate=0.15, 
                 dist_threhold=1000):
        super(MDN_Block, self).__init__()
        self.MLP = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ELU(), nn.Dropout(p=dropout_rate)) 
        self.z_pi = nn.Linear(hidden_dim, n_gaussians)
        self.z_sigma = nn.Linear(hidden_dim, n_gaussians)
        self.z_mu = nn.Linear(hidden_dim, n_gaussians)
        self.atom_types = nn.Linear(hidden_dim, 18)
        self.bond_types = nn.Linear(hidden_dim*2, 5)        
        self.dist_threhold = dist_threhold
        
        # 新增：自适应阈值预测
        self.threshold_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )
        
        # 新增：特征增强模块
        self.feature_enhancer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim * 2)
        )
    
    def forward(self, loop_s, loop_pos, loop_batch, pro_s, pro_pos, pro_batch, edge_index):
        
        h_l_x, l_mask = to_dense_batch(loop_s, loop_batch, fill_value=0)
        h_t_x, t_mask = to_dense_batch(pro_s, pro_batch, fill_value=0)
        h_l_pos, _ = to_dense_batch(loop_pos, loop_batch, fill_value=0)
        h_t_pos, _ = to_dense_batch(pro_pos, pro_batch, fill_value=0)
        
        assert h_l_x.size(0) == h_t_x.size(0), 'Encountered unequal batch-sizes'
        (B, N_l, C_out), N_t = h_l_x.size(), h_t_x.size(1)
        self.B = B
        self.N_l = N_l
        self.N_t = N_t
        # Combine and mask
        h_l_x = h_l_x.unsqueeze(-2)
        h_l_x = h_l_x.repeat(1, 1, N_t, 1) # [B, N_l, N_t, C_out]

        h_t_x = h_t_x.unsqueeze(-3)
        h_t_x = h_t_x.repeat(1, N_l, 1, 1) # [B, N_l, N_t, C_out]

        C = torch.cat((h_l_x, h_t_x), -1)
        
        C = C + self.feature_enhancer(C)  # 残差连接
        
        self.C_mask = C_mask = l_mask.view(B, N_l, 1) & t_mask.view(B, 1, N_t)
        self.C = C = C[C_mask]
        C = self.MLP(C)

        # Get batch indexes for ligand-target combined features
        C_batch = torch.tensor(range(B)).unsqueeze(-1).unsqueeze(-1).to(loop_s.device)
        C_batch = C_batch.repeat(1, N_l, N_t)[C_mask]
        
        # Outputs
        pi = F.softmax(self.z_pi(C), -1)
        sigma = F.elu(self.z_sigma(C))+1.1
        mu = F.elu(self.z_mu(C))+1
        dist = self.compute_euclidean_distances_matrix(h_l_pos, h_t_pos.view(h_t_pos.size(0), -1, 3))[C_mask]
        atom_types = self.atom_types(loop_s)
        bond_types = self.bond_types(torch.cat([loop_s[edge_index[0]],loop_s[edge_index[1]]], axis=1))
        return pi, sigma, mu, dist.unsqueeze(1).detach(), C_batch, atom_types, bond_types






# class MDN_Block(nn.Module):
#     def __init__(self, hidden_dim, n_gaussians, dropout_rate=0.15, 
#                  dist_threhold=1000):
#         super(MDN_Block, self).__init__()
#         self.MLP = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ELU(), nn.Dropout(p=dropout_rate)) 
#         self.z_pi = nn.Linear(hidden_dim, n_gaussians)
#         self.z_sigma = nn.Linear(hidden_dim, n_gaussians)
#         self.z_mu = nn.Linear(hidden_dim, n_gaussians)
#         self.atom_types = nn.Linear(hidden_dim, 18)
#         self.bond_types = nn.Linear(hidden_dim*2, 5)        
#         self.dist_threhold = dist_threhold
        
#         # 新增：自适应阈值预测
#         self.threshold_predictor = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.SiLU(),
#             nn.Linear(hidden_dim // 2, 1),
#             nn.Softplus()
#         )
        
#         self.feature_enhancer = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim * 2),
#             nn.LayerNorm(hidden_dim * 2),
#             nn.SiLU(),
#             nn.Dropout(dropout_rate),
#             nn.Linear(hidden_dim * 2, hidden_dim * 2)
#         )
        
#         self.attention_gate = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 4),
#             nn.SiLU(),
#             nn.Linear(hidden_dim // 4, 1),
#             nn.Sigmoid()
#         )
            
#     def forward(self, loop_s, loop_pos, loop_batch, pro_s, pro_pos, pro_batch, edge_index):
        
#         h_l_x, l_mask = to_dense_batch(loop_s, loop_batch, fill_value=0)
#         h_t_x, t_mask = to_dense_batch(pro_s, pro_batch, fill_value=0)
#         h_l_pos, _ = to_dense_batch(loop_pos, loop_batch, fill_value=0)
#         h_t_pos, _ = to_dense_batch(pro_pos, pro_batch, fill_value=0)
        
#         assert h_l_x.size(0) == h_t_x.size(0), 'Encountered unequal batch-sizes'
#         (B, N_l, C_out), N_t = h_l_x.size(), h_t_x.size(1)
#         self.B = B
#         self.N_l = N_l
#         self.N_t = N_t
#         # Combine and mask
#         h_l_x = h_l_x.unsqueeze(-2)
#         h_l_x = h_l_x.repeat(1, 1, N_t, 1) # [B, N_l, N_t, C_out]

#         h_t_x = h_t_x.unsqueeze(-3)
#         h_t_x = h_t_x.repeat(1, N_l, 1, 1) # [B, N_l, N_t, C_out]

#         C = torch.cat((h_l_x, h_t_x), -1)

#         C = C + self.feature_enhancer(C)  # 残差连接
        
#         self.C_mask = C_mask = l_mask.view(B, N_l, 1) & t_mask.view(B, 1, N_t)
#         self.C = C = C[C_mask]
#         C_features  = self.MLP(C)

#         # Get batch indexes for ligand-target combined features
#         C_batch = torch.tensor(range(B)).unsqueeze(-1).unsqueeze(-1).to(loop_s.device)
#         C_batch = C_batch.repeat(1, N_l, N_t)[C_mask]
        
#         # 计算距离（用于损失计算）
#         dist = self.compute_euclidean_distances_matrix(h_l_pos, h_t_pos.view(h_t_pos.size(0), -1, 3))[C_mask]
        
#         # 【改进1】使用自适应阈值
#         adaptive_thresholds = self.threshold_predictor(C_features) + 5.0  # [N_valid, 1] 基础阈值5.0
        
#         # 计算距离掩码（软掩码）
#         dist_mask = torch.sigmoid((adaptive_thresholds - dist.unsqueeze(1)) / 2.0)  # [N_valid, 1]
        
#         # 【改进3】计算注意力权重
#         attention_weights = self.attention_gate(C_features)  # [N_valid, 1]
        
#         # MDN输出（结合注意力权重和距离掩码）
#         pi_logits = self.z_pi(C_features)
#         # 距离越近，权重越大
#         pi_logits = pi_logits + dist_mask.expand_as(pi_logits).squeeze(1)
#         pi = F.softmax(pi_logits, -1)
        
#         # 应用注意力权重到其他参数
#         sigma = (F.elu(self.z_sigma(C_features)) + 1.1) * attention_weights
#         mu = (F.elu(self.z_mu(C_features)) + 1) * attention_weights
        
#         # 原子和键类型预测
#         atom_types = self.atom_types(loop_s)
#         bond_types = self.bond_types(torch.cat([loop_s[edge_index[0]], loop_s[edge_index[1]]], axis=1))
        
#         # 返回所有结果
#         return pi, sigma.squeeze(-1), mu.squeeze(-1), dist.unsqueeze(1).detach(), C_batch, atom_types, bond_types
    


    # # 原始
    # def __init__(self, hidden_dim, n_gaussians, dropout_rate=0.15, 
    #              dist_threhold=1000):
    #     super(MDN_Block, self).__init__()
    #     self.MLP = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ELU(), nn.Dropout(p=dropout_rate)) 
    #     self.z_pi = nn.Linear(hidden_dim, n_gaussians)
    #     self.z_sigma = nn.Linear(hidden_dim, n_gaussians)
    #     self.z_mu = nn.Linear(hidden_dim, n_gaussians)
    #     self.atom_types = nn.Linear(hidden_dim, 18)
    #     self.bond_types = nn.Linear(hidden_dim*2, 5)        
    #     self.dist_threhold = dist_threhold
    
    # def forward(self, loop_s, loop_pos, loop_batch, pro_s, pro_pos, pro_batch, edge_index):
        
    #     h_l_x, l_mask = to_dense_batch(loop_s, loop_batch, fill_value=0)
    #     h_t_x, t_mask = to_dense_batch(pro_s, pro_batch, fill_value=0)
    #     h_l_pos, _ = to_dense_batch(loop_pos, loop_batch, fill_value=0)
    #     h_t_pos, _ = to_dense_batch(pro_pos, pro_batch, fill_value=0)
        
    #     assert h_l_x.size(0) == h_t_x.size(0), 'Encountered unequal batch-sizes'
    #     (B, N_l, C_out), N_t = h_l_x.size(), h_t_x.size(1)
    #     self.B = B
    #     self.N_l = N_l
    #     self.N_t = N_t
    #     # Combine and mask
    #     h_l_x = h_l_x.unsqueeze(-2)
    #     h_l_x = h_l_x.repeat(1, 1, N_t, 1) # [B, N_l, N_t, C_out]

    #     h_t_x = h_t_x.unsqueeze(-3)
    #     h_t_x = h_t_x.repeat(1, N_l, 1, 1) # [B, N_l, N_t, C_out]

    #     C = torch.cat((h_l_x, h_t_x), -1)
        
    #     self.C_mask = C_mask = l_mask.view(B, N_l, 1) & t_mask.view(B, 1, N_t)
    #     self.C = C = C[C_mask]
    #     C = self.MLP(C)

    #     # Get batch indexes for ligand-target combined features
    #     C_batch = torch.tensor(range(B)).unsqueeze(-1).unsqueeze(-1).to(loop_s.device)
    #     C_batch = C_batch.repeat(1, N_l, N_t)[C_mask]
        
    #     # Outputs
    #     pi = F.softmax(self.z_pi(C), -1)
    #     sigma = F.elu(self.z_sigma(C))+1.1
    #     mu = F.elu(self.z_mu(C))+1
    #     dist = self.compute_euclidean_distances_matrix(h_l_pos, h_t_pos.view(h_t_pos.size(0), -1, 3))[C_mask]
    #     atom_types = self.atom_types(loop_s)
    #     bond_types = self.bond_types(torch.cat([loop_s[edge_index[0]],loop_s[edge_index[1]]], axis=1))
    #     return pi, sigma, mu, dist.unsqueeze(1).detach(), C_batch, atom_types, bond_types



    def compute_euclidean_distances_matrix(self, X, Y):
        # Based on: https://medium.com/@souravdey/l2-distance-matrix-vectorization-trick-26aa3247ac6c
        # (X-Y)^2 = X^2 + Y^2 -2XY
        X = X.double()
        Y = Y.double()

        dists = -2 * torch.bmm(X, Y.permute(0, 2, 1)) + torch.sum(Y**2, axis=-1).unsqueeze(1) + torch.sum(X**2, axis=-1).unsqueeze(-1)
        # return dists**0.5
        return torch.nan_to_num((dists**0.5).view(self.B, self.N_l,-1,24),10000).min(axis=-1)[0]

    
    def mdn_loss_fn(self, pi, sigma, mu, y):
        epsilon = 1e-16
        normal = Normal(mu, sigma+epsilon)
        loglik = normal.log_prob(y.expand_as(normal.loc)) + epsilon
        pi = torch.softmax(pi, dim=1)
        loss = -torch.logsumexp(torch.log(pi) + loglik, dim=1)
        return loss
    
    def calculate_probablity(self, pi, sigma, mu, y):
        normal = Normal(mu, sigma)
        logprob = normal.log_prob(y.expand_as(normal.loc))
        logprob += torch.log(pi)
        prob = logprob.exp().sum(1)        
        return prob
