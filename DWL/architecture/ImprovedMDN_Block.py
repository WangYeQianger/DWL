#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   Improved MDN Block
@Time    :   2024/01/10
@Author  :   Enhanced version with innovative modules
@Version :   2.0
@Desc    :   改进的MDN模块，包含注意力机制和自适应阈值
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch_geometric.utils import to_dense_batch


class ImprovedMDN_Block(nn.Module):
    def __init__(self, hidden_dim, n_gaussians, dropout_rate=0.15, 
                 dist_threshold=1000, use_attention=True):
        super(ImprovedMDN_Block, self).__init__()
        
        # 原始组件
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim), 
            nn.BatchNorm1d(hidden_dim), 
            nn.ELU(), 
            nn.Dropout(p=dropout_rate)
        )
        
        # MDN 输出头
        self.z_pi = nn.Linear(hidden_dim, n_gaussians)
        self.z_sigma = nn.Linear(hidden_dim, n_gaussians)
        self.z_mu = nn.Linear(hidden_dim, n_gaussians)
        self.atom_types = nn.Linear(hidden_dim, 18)
        self.bond_types = nn.Linear(hidden_dim*2, 5)
        
        self.dist_threshold = dist_threshold
        self.use_attention = use_attention
        
        # 创新1：自适应阈值预测器（改进版）
        # 不仅预测阈值，还预测每对原子的重要性权重
        self.adaptive_threshold = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 2)  # 输出：[阈值, 重要性权重]
        )
        
        # 创新2：交叉注意力特征增强器
        # 使用注意力机制来增强配体-蛋白质交互特征
        if use_attention:
            self.cross_attention = CrossAttentionEnhancer(hidden_dim, dropout_rate)
        
        # 创新3：空间感知模块
        # 整合空间信息到特征表示中
        self.spatial_encoder = SpatialAwareModule(hidden_dim)
        
        # 创新4：不确定性估计模块
        # 预测模型对每个预测的置信度
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, loop_s, loop_pos, loop_batch, pro_s, pro_pos, pro_batch, edge_index):
        # 转换为密集批次
        h_l_x, l_mask = to_dense_batch(loop_s, loop_batch, fill_value=0)
        h_t_x, t_mask = to_dense_batch(pro_s, pro_batch, fill_value=0)
        h_l_pos, _ = to_dense_batch(loop_pos, loop_batch, fill_value=0)
        h_t_pos, _ = to_dense_batch(pro_pos, pro_batch, fill_value=0)
        
        assert h_l_x.size(0) == h_t_x.size(0), 'Encountered unequal batch-sizes'
        (B, N_l, C_out), N_t = h_l_x.size(), h_t_x.size(1)
        self.B = B
        self.N_l = N_l
        self.N_t = N_t
        
        # 应用空间感知编码
        h_l_x = self.spatial_encoder(h_l_x, h_l_pos)
        h_t_x = self.spatial_encoder(h_t_x, h_t_pos)
        
        # 扩展维度以计算所有配对
        h_l_x_expanded = h_l_x.unsqueeze(-2).repeat(1, 1, N_t, 1)
        h_t_x_expanded = h_t_x.unsqueeze(-3).repeat(1, N_l, 1, 1)
        
        # 连接特征
        C = torch.cat((h_l_x_expanded, h_t_x_expanded), -1)
        
        # 应用交叉注意力增强（如果启用）
        if self.use_attention:
            C = self.cross_attention(C, h_l_x, h_t_x, l_mask, t_mask)
        
        # 创建掩码
        self.C_mask = C_mask = l_mask.view(B, N_l, 1) & t_mask.view(B, 1, N_t)
        self.C = C = C[C_mask]
        
        # 通过MLP处理
        C_features = self.MLP(C)
        
        # 获取批次索引
        C_batch = torch.tensor(range(B)).unsqueeze(-1).unsqueeze(-1).to(loop_s.device)
        C_batch = C_batch.repeat(1, N_l, N_t)[C_mask]
        
        # 计算距离
        dist = self.compute_euclidean_distances_matrix(h_l_pos, h_t_pos.view(h_t_pos.size(0), -1, 3))[C_mask]
        
        # 使用自适应阈值和重要性权重
        adaptive_output = self.adaptive_threshold(C)
        adaptive_threshold = F.softplus(adaptive_output[:, 0:1]) + 10.0  # 确保阈值为正
        importance_weights = torch.sigmoid(adaptive_output[:, 1:2])
        
        # 应用自适应阈值进行距离过滤
        dist_filtered = torch.where(
            dist < adaptive_threshold,
            dist,
            torch.full_like(dist, self.dist_threshold)
        )
        
        # MDN 输出
        pi = F.softmax(self.z_pi(C_features), -1)
        sigma = F.elu(self.z_sigma(C_features)) + 1.1
        mu = F.elu(self.z_mu(C_features)) + 1
        
        # 不确定性估计
        uncertainty = self.uncertainty_estimator(C_features)
        
        # 原子和键类型预测
        atom_types = self.atom_types(loop_s)
        bond_types = self.bond_types(torch.cat([loop_s[edge_index[0]], loop_s[edge_index[1]]], axis=1))
        
        # 返回增强的输出
        return {
            'pi': pi,
            'sigma': sigma,
            'mu': mu,
            'dist': dist_filtered.unsqueeze(1).detach(),
            'C_batch': C_batch,
            'atom_types': atom_types,
            'bond_types': bond_types,
            'importance_weights': importance_weights,
            'uncertainty': uncertainty,
            'adaptive_threshold': adaptive_threshold
        }
    
    def compute_euclidean_distances_matrix(self, X, Y):
        X = X.double()
        Y = Y.double()
        dists = -2 * torch.bmm(X, Y.permute(0, 2, 1)) + torch.sum(Y**2, axis=-1).unsqueeze(1) + torch.sum(X**2, axis=-1).unsqueeze(-1)
        return torch.nan_to_num((dists**0.5).view(self.B, self.N_l, -1, 24), 10000).min(axis=-1)[0]
    
    def mdn_loss_fn(self, pi, sigma, mu, y, importance_weights=None, uncertainty=None):
        epsilon = 1e-16
        normal = Normal(mu, sigma + epsilon)
        loglik = normal.log_prob(y.expand_as(normal.loc)) + epsilon
        pi = torch.softmax(pi, dim=1)
        loss = -torch.logsumexp(torch.log(pi) + loglik, dim=1)
        
        # 应用重要性权重
        if importance_weights is not None:
            loss = loss * importance_weights.squeeze()
        
        # 应用不确定性权重（高不确定性 = 低权重）
        if uncertainty is not None:
            loss = loss * (1 - uncertainty.squeeze() * 0.5)  # 最多降低50%权重
        
        return loss
    
    def calculate_probability(self, pi, sigma, mu, y):
        normal = Normal(mu, sigma)
        logprob = normal.log_prob(y.expand_as(normal.loc))
        logprob += torch.log(pi)
        prob = logprob.exp().sum(1)
        return prob


class CrossAttentionEnhancer(nn.Module):
    """交叉注意力模块，增强配体-蛋白质交互特征"""
    def __init__(self, hidden_dim, dropout_rate=0.15, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        
        # 多头注意力投影
        self.q_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
    def forward(self, pair_features, ligand_features, protein_features, l_mask, p_mask):
        B, N_l, N_p, C = pair_features.shape
        
        # 重塑pair features用于注意力计算
        pair_flat = pair_features.view(B, N_l * N_p, C)
        
        # 计算查询（从配对特征）
        Q = self.q_proj(pair_flat).view(B, N_l * N_p, self.num_heads, self.head_dim)
        Q = Q.transpose(1, 2)  # [B, heads, N_l*N_p, head_dim]
        
        # 计算键和值（从配体和蛋白质特征）
        all_features = torch.cat([ligand_features, protein_features], dim=1)  # [B, N_l+N_p, C]
        K = self.k_proj(all_features).view(B, N_l + N_p, self.num_heads, self.head_dim)
        V = self.v_proj(all_features).view(B, N_l + N_p, self.num_heads, self.head_dim)
        K = K.transpose(1, 2)  # [B, heads, N_l+N_p, head_dim]
        V = V.transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # 应用掩码
        all_mask = torch.cat([l_mask, p_mask], dim=1)  # [B, N_l+N_p]
        mask_expanded = all_mask.unsqueeze(1).unsqueeze(1).expand(-1, self.num_heads, N_l * N_p, -1)
        scores = scores.masked_fill(~mask_expanded, -1e9)
        
        # Softmax和应用注意力
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)  # [B, heads, N_l*N_p, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N_l * N_p, self.hidden_dim)
        
        # 输出投影并重塑
        output = self.o_proj(attn_output).view(B, N_l, N_p, C)
        
        # 残差连接和层归一化
        output = self.layer_norm(output + pair_features)
        
        return output


class SpatialAwareModule(nn.Module):
    """空间感知模块，将3D坐标信息整合到特征中"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.spatial_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2)
        )
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, features, positions):
        # 编码空间位置
        spatial_features = self.spatial_encoder(positions)
        
        # 融合空间和原始特征
        combined = torch.cat([features, spatial_features], dim=-1)
        output = self.feature_fusion(combined)
        
        # 残差连接
        return output + features