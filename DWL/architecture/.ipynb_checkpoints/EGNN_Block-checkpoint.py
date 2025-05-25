#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   EGNN Block
@Time    :   2022/09/05 10:57:54
@Author  :   Xujun Zhang, Tianyue Wang
@Version :   1.0
@Contact :   
@License :   
@Desc    :   None
'''

# here put the import lib
import pandas as pd
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GraphNorm
from torch_geometric.utils import softmax, to_dense_batch
from torch_scatter import scatter, scatter_mean

# class EGNN(nn.Module):
    # def __init__(self, dim_in, dim_tmp, edge_in, edge_out, num_head=8, drop_rate=0.15, update_pos=False):
    #     super().__init__()
    #     assert dim_tmp % num_head == 0
    #     self.edge_dim = edge_in
    #     self.num_head = num_head # 4
    #     self.dh = dim_tmp // num_head # 32
    #     self.dim_tmp = dim_tmp # 12
    #     self.q_layer = nn.Linear(dim_in, dim_tmp)
    #     self.k_layer = nn.Linear(dim_in, dim_tmp)
    #     self.v_layer = nn.Linear(dim_in, dim_tmp)
    #     self.m_layer = nn.Sequential(
    #         nn.Linear(edge_in+1, dim_tmp),
    #         nn.Dropout(p=drop_rate),
    #         nn.LeakyReLU(), 
    #         nn.Linear(dim_tmp, dim_tmp)
    #         )
    #     self.m2f_layer = nn.Sequential(
    #         nn.Linear(dim_tmp, dim_tmp),
    #         nn.Dropout(p=drop_rate))
    #     self.e_layer = nn.Sequential(
    #         nn.Linear(dim_tmp, edge_out),
    #         nn.Dropout(p=drop_rate))
    #     self.gate_layer = nn.Sequential(
    #         nn.Linear(3*dim_tmp, dim_tmp),
    #         nn.Dropout(p=drop_rate))
    #     self.layer_norm_1 = GraphNorm(dim_tmp)
    #     self.layer_norm_2 = GraphNorm(dim_tmp)
    #     self.fin_layer = nn.Sequential(
    #         nn.Linear(dim_tmp, dim_tmp),
    #         nn.Dropout(p=drop_rate),
    #         nn.LeakyReLU(),
    #         nn.Linear(dim_tmp, dim_tmp)
    #         )
    #     self.update_pos = update_pos
    #     if update_pos:
    #         self.update_layer = coords_update(dim_dh=self.dh, num_head=num_head, drop_rate=drop_rate)
    
    # def forward(self, node_s, edge_s, edge_index, generate_node_dist, pos, parent_node_idxes, generate_node_idxes, mask_edge_inv, pro_nodes_num, batch):
    
    #     q_ = self.q_layer(node_s)
    #     k_ = self.k_layer(node_s)
    #     v_ = self.v_layer(node_s)
    #     # message passing
    #     ## cal distance
    #     d_ij = torch.pairwise_distance(pos[edge_index[0]], pos[edge_index[1]]).unsqueeze(dim=-1)*0.1
    #     ## mask unkonwn distance    
    #     d_ij[mask_edge_inv] = -1
    #     ## cal attention
    #     m_ij = torch.cat([edge_s, d_ij], dim=-1)
    #     m_ij = self.m_layer(m_ij)
    #     k_ij = k_[edge_index[1]] * m_ij
    #     a_ij = ((q_[edge_index[0]] * k_ij)/math.sqrt(self.dh)).view((-1, self.num_head, self.dh))
        
        
    #     # Zip the lists and the a_ij variable together
    #     w_ij = softmax(torch.norm(a_ij, p=1, dim=2), index=edge_index[0]).unsqueeze(dim=-1)
    #     # update node and edge embeddings 
    #     node_s_new = self.m2f_layer(scatter(w_ij*v_[edge_index[1]].view((-1, self.num_head, self.dh)), index=edge_index[0], reduce='sum', dim=0).view((-1, self.dim_tmp)))
    #     edge_s_new = self.e_layer(a_ij.view((-1, self.dim_tmp)))
    #     g = torch.sigmoid(self.gate_layer(torch.cat([node_s_new, node_s, node_s_new-node_s], dim=-1)))
    #     node_s_new = self.layer_norm_1(g*node_s_new+node_s, batch)
    #     node_s_new = self.layer_norm_2(g*self.fin_layer(node_s_new)+node_s_new, batch)
    #     # update coords
    #     if self.update_pos:
    #         delta_x = self.update_layer(a_ij, pos, generate_node_dist, edge_index, parent_node_idxes, generate_node_idxes, mask_edge_inv, pro_nodes_num)
    #     else:
    #         delta_x = None
    #     return node_s_new, edge_s_new, delta_x





class EGNN(nn.Module):
    def __init__(self, dim_in, dim_tmp, edge_in, edge_out, num_head=8, drop_rate=0.15, update_pos=False):
        super().__init__()
        assert dim_tmp % num_head == 0
        self.edge_dim = edge_in
        self.num_head = num_head # 4
        self.dh = dim_tmp // num_head # 32
        self.dim_tmp = dim_tmp # 12
        self.q_layer = nn.Linear(dim_in, dim_tmp)
        self.k_layer = nn.Linear(dim_in, dim_tmp)
        self.v_layer = nn.Linear(dim_in, dim_tmp)
        self.m_layer = nn.Sequential(
            nn.Linear(edge_in+1, dim_tmp),
            nn.Dropout(p=drop_rate),
            nn.LeakyReLU(), 
            nn.Linear(dim_tmp, dim_tmp)
            )
        self.m2f_layer = nn.Sequential(
            nn.Linear(dim_tmp, dim_tmp),
            nn.Dropout(p=drop_rate))
        self.e_layer = nn.Sequential(
            nn.Linear(dim_tmp, edge_out),
            nn.Dropout(p=drop_rate))
        self.gate_layer = nn.Sequential(
            nn.Linear(3*dim_tmp, dim_tmp),
            nn.Dropout(p=drop_rate))
        self.layer_norm_1 = GraphNorm(dim_tmp)
        self.layer_norm_2 = GraphNorm(dim_tmp)
        self.fin_layer = nn.Sequential(
            nn.Linear(dim_tmp, dim_tmp),
            nn.Dropout(p=drop_rate),
            nn.LeakyReLU(),
            nn.Linear(dim_tmp, dim_tmp)
            )
        self.update_pos = update_pos
        if update_pos:
            self.update_layer = coords_update(dim_dh=self.dh, num_head=num_head, drop_rate=drop_rate)
            
            
        # 新增：几何特征提取器
        self.geom_feat_extractor = nn.Sequential(
            nn.Linear(4, dim_tmp),  # 输入: [距离, 角度余弦, 二面角余弦, 是否相邻]
            nn.SiLU(),
            nn.Linear(dim_tmp, dim_tmp)
        )
        
        # 新增：位置敏感的注意力调制
        self.pos_attention_gate = nn.Sequential(
            nn.Linear(dim_tmp + 1, dim_tmp),
            nn.SiLU(),
            nn.Linear(dim_tmp, num_head)
        )


    def compute_geometric_features(self, pos, edge_index, batch):
        """计算几何特征（保持 O(E) 复杂度）"""
        row, col = edge_index
        # 边向量 & 距离
        edge_vec  = pos[col] - pos[row]                                 # [E,3]
        edge_dist = torch.norm(edge_vec, dim=-1, keepdim=True)          # [E,1]

        # 单位向量
        edge_unit = edge_vec / (edge_dist + 1e-8)                       # [E,3]

        num_nodes = pos.size(0)
        device    = pos.device
        dtype3    = edge_unit.dtype

        # 1) 聚合每个节点的“出边”单位向量之和
        sum_src = torch.zeros((num_nodes, 3), device=device, dtype=dtype3)
        sum_src.index_add_(0, row, edge_unit)                          # [N,3]
        sum_dst = torch.zeros((num_nodes, 3), device=device, dtype=dtype3)
        sum_dst.index_add_(0, col, edge_unit)                          # [N,3]

        # 2) 聚合每个节点的度（用来判断是否有相邻边）
        deg_src = torch.zeros((num_nodes,), device=device, dtype=edge_dist.dtype)
        deg_src.index_add_(0, row, torch.ones_like(row, dtype=deg_src.dtype))  
        deg_dst = torch.zeros((num_nodes,), device=device, dtype=edge_dist.dtype)
        deg_dst.index_add_(0, col, torch.ones_like(col, dtype=deg_dst.dtype))

        # 对应到每条边上，去掉当前边自身后的邻居信息
        neigh_src_sum = sum_src[row] - edge_unit                       # [E,3]
        neigh_dst_sum = sum_dst[col] - edge_unit                       # [E,3]

        # 计算 angle_cos = ⟨u, sum(neighbors)⟩ / (‖sum(neighbors)‖ + ε)
        dot_src   = (edge_unit * neigh_src_sum).sum(dim=-1)             # [E]
        norm_src  = neigh_src_sum.norm(dim=-1)                          # [E]
        angle_cos = (dot_src / (norm_src + 1e-8)).unsqueeze(-1)         # [E,1]

        # 计算 dihedral_cos：用两个叉积的法向量之间的夹角
        normal1      = torch.cross(edge_unit, neigh_src_sum, dim=-1)   # [E,3]
        normal2      = torch.cross(edge_unit, neigh_dst_sum, dim=-1)   # [E,3]
        n1           = normal1.norm(dim=-1)                            # [E]
        n2           = normal2.norm(dim=-1)                            # [E]
        dihedral_cos = ((normal1 * normal2).sum(dim=-1) / (n1 * n2 + 1e-8)).unsqueeze(-1)

        # 计算 is_adjacent：若边的任一端节点度 > 1，则视为“有相邻边”
        is_adj = ((deg_src[row] > 1) | (deg_dst[col] > 1)).unsqueeze(-1).to(edge_dist.dtype)  # [E,1]

        # 拼接所有几何特征
        geom_feats = torch.cat([
            edge_dist / 10.0,    # 归一化距离
            angle_cos,           # 角度余弦
            dihedral_cos,        # 二面角余弦
            is_adj               # 相邻性指示
        ], dim=-1)              # [E, 4]

        return geom_feats




    # def compute_geometric_features(self, pos, edge_index, batch):
    #     """计算几何特征"""
    #     row, col = edge_index
        
    #     # 边向量
    #     edge_vec = pos[col] - pos[row]
    #     edge_dist = torch.norm(edge_vec, dim=-1, keepdim=True)
        
    #     # 计算角度（需要三个连续的节点）
    #     # 这里简化处理，实际可以通过图结构找到三元组
    #     angle_cos = torch.zeros_like(edge_dist)
    #     dihedral_cos = torch.zeros_like(edge_dist)
    #     is_adjacent = torch.zeros_like(edge_dist)
        
    #     # 组合几何特征
    #     geom_feats = torch.cat([
    #         edge_dist / 10.0,  # 归一化距离
    #         angle_cos,
    #         dihedral_cos,
    #         is_adjacent
    #     ], dim=-1)
        
        
    #     # geom_feats = edge_dist / 10.0
        
    #     return geom_feats
    
    
    def forward(self, node_s, edge_s, edge_index, generate_node_dist, pos, parent_node_idxes, generate_node_idxes, mask_edge_inv, pro_nodes_num, batch):
        # 计算几何特征
        geom_feats = self.compute_geometric_features(pos, edge_index, batch)
        geom_embed = self.geom_feat_extractor(geom_feats)
        
        
        # print(geom_feats[0])
        # print(geom_feats.shape)
        
        # geom_embed = geom_feats
                
        # print(edge_s.shape)
        # print(geom_embed.shape)
                
        # 融合几何特征到边特征
        edge_s = edge_s + 0.1 * geom_embed
        
        
        q_ = self.q_layer(node_s)
        k_ = self.k_layer(node_s)
        v_ = self.v_layer(node_s)
        # message passing
        ## cal distance
        d_ij = torch.pairwise_distance(pos[edge_index[0]], pos[edge_index[1]]).unsqueeze(dim=-1)*0.1
        ## mask unkonwn distance    
        d_ij[mask_edge_inv] = -1
        ## cal attention
        m_ij = torch.cat([edge_s, d_ij], dim=-1)
        m_ij = self.m_layer(m_ij)
        k_ij = k_[edge_index[1]] * m_ij
        a_ij = ((q_[edge_index[0]] * k_ij)/math.sqrt(self.dh)).view((-1, self.num_head, self.dh))
        
        
        # 使用位置敏感的注意力门控
        d_ij = torch.pairwise_distance(pos[edge_index[0]], pos[edge_index[1]]).unsqueeze(dim=-1)*0.1
        pos_gate_input = torch.cat([m_ij, d_ij], dim=-1)
        pos_gate = torch.sigmoid(self.pos_attention_gate(pos_gate_input))
        
        # 调制注意力权重
        a_ij = a_ij * pos_gate.unsqueeze(-1)
        
        
        # Zip the lists and the a_ij variable together
        w_ij = softmax(torch.norm(a_ij, p=1, dim=2), index=edge_index[0]).unsqueeze(dim=-1)
        # update node and edge embeddings 
        node_s_new = self.m2f_layer(scatter(w_ij*v_[edge_index[1]].view((-1, self.num_head, self.dh)), index=edge_index[0], reduce='sum', dim=0).view((-1, self.dim_tmp)))
        edge_s_new = self.e_layer(a_ij.view((-1, self.dim_tmp)))
        g = torch.sigmoid(self.gate_layer(torch.cat([node_s_new, node_s, node_s_new-node_s], dim=-1)))
        node_s_new = self.layer_norm_1(g*node_s_new+node_s, batch)
        node_s_new = self.layer_norm_2(g*self.fin_layer(node_s_new)+node_s_new, batch)
        # update coords
        if self.update_pos:
            delta_x = self.update_layer(a_ij, pos, generate_node_dist, edge_index, parent_node_idxes, generate_node_idxes, mask_edge_inv, pro_nodes_num)
        else:
            delta_x = None
        return node_s_new, edge_s_new, delta_x






class coords_update(nn.Module):
    def __init__(self, dim_dh, num_head, drop_rate=0.15):
        super().__init__()
        # self.correct_clash_num = correct_clash_num
        self.num_head = num_head
        self.attention2deltax = nn.Sequential(
            nn.Linear(dim_dh, dim_dh//2),
            nn.Dropout(p=drop_rate),
            nn.LeakyReLU(),
            nn.Linear(dim_dh//2, 1)
        )
        self.weighted_head_layer = nn.Linear(num_head, 1, bias=False)
        

    def forward(self, a_ij, pos, generate_node_dist, edge_index, parent_node_idxes, generate_node_idxes, mask_edge_inv, pro_nodes_num):
        edge_index_mask = torch.isin(edge_index[0], generate_node_idxes)
        mask_edge_inv = mask_edge_inv.squeeze(dim=-1)[edge_index_mask]
        edge_index = edge_index[:, edge_index_mask]
        a_ij = a_ij[edge_index_mask, :, :]
        # cal vector delta_x
        delta_x = pos[edge_index[0]] - pos[edge_index[1]]
        delta_x = delta_x/(torch.norm(delta_x, p=2, dim=-1).unsqueeze(dim=-1) + 1e-6 )
        # mask distance calculated by node with unknown coords
        delta_x[mask_edge_inv] = torch.zeros_like(delta_x[mask_edge_inv])
        # cal attention
        attention= self.weighted_head_layer(self.attention2deltax(a_ij).squeeze(dim=2))
        delta_x = delta_x*attention
        delta_x = scatter(delta_x, index=edge_index[0], reduce='sum', dim=0, dim_size=pos.size(0))
        # parent_node_idxes
        delta_x = delta_x[generate_node_idxes, :]
        # normalize delta_x
        # if (parent_node_idxes < pro_nodes_num).all():
        #     d_ij = torch.norm(delta_x, p=2, dim=-1).unsqueeze(dim=-1) + 1e-6
        # else:
        #     d_ij = generate_node_dist
        d_ij = generate_node_dist    
        delta_x = delta_x/(torch.norm(delta_x, p=2, dim=-1).unsqueeze(dim=-1) + 1e-6 ) * d_ij 
        # update coords
        # pos += delta_x
        # pos_new = pos.clone()
        # pos_new[generate_node_idxes,:] = pos[parent_node_idxes, : ] + delta_x
        # return pos[generate_node_idxes, : ] + delta_x
        
        return delta_x


# class coords_update(nn.Module):
#     def __init__(self, dim_dh, num_head, drop_rate=0.15):
#         super().__init__()
#         self.num_head = num_head

#         # 将每个 head 的 dh 维特征映射成 Δx 的 attention 分数
#         self.attention2deltax = nn.Sequential(
#             nn.Linear(dim_dh, dim_dh // 2),
#             nn.Dropout(p=drop_rate),
#             nn.LeakyReLU(),
#             nn.Linear(dim_dh // 2, 1)
#         )
#         # 将 num_head 个 attention 分数融合为一个标量
#         self.weighted_head_layer = nn.Linear(num_head, 1, bias=False)

#         # —— 新增：学习残差缩放因子 ——
#         self.residual_gate = nn.Sequential(
#             nn.Linear(dim_dh * num_head, dim_dh),
#             nn.SiLU(),
#             nn.Linear(dim_dh, 1),
#             nn.Sigmoid()
#         )
#         # —— 新增：距离预测器 ——
#         self.distance_predictor = nn.Sequential(
#             nn.Linear(dim_dh * num_head, dim_dh),
#             nn.SiLU(),
#             nn.Linear(dim_dh, 1),
#             nn.Softplus()  # 保证输出正值
#         )

#     def forward(self,
#                 a_ij,
#                 pos,
#                 generate_node_dist,
#                 edge_index,
#                 parent_node_idxes,
#                 generate_node_idxes,
#                 mask_edge_inv,
#                 pro_nodes_num):
#         # 只对“待生成”节点相关的边做更新
#         mask_e = torch.isin(edge_index[0], generate_node_idxes)  # [E]
#         ei = edge_index[:, mask_e]                              # [2, E']
#         a = a_ij[mask_e]                                        # [E', num_head, dh]

#         E = a.size(0)

#         # 将多头特征扁平化，用于预测距离 & 残差门控
#         a_flat = a.view(E, -1)                                  # [E', num_head*dh]
#         pred_d = self.distance_predictor(a_flat)                # [E',1]

#         # 1) 方向向量归一化
#         raw_dx = pos[ei[0]] - pos[ei[1]]                        # [E',3]
#         cur_d = torch.norm(raw_dx, p=2, dim=-1, keepdim=True).clamp_min(1e-6)  # [E',1]
#         unit_dx = raw_dx / cur_d                                # [E',3]

#         # 屏蔽无效边
#         me = mask_edge_inv.squeeze(-1)[mask_e]                  # [E']
#         unit_dx[me] = 0

#         # 2) 拉伸到预测长度
#         dx = unit_dx * pred_d                                   # [E',3]

#         # 3) attention 分数
#         #    a.view → [E'*num_head, dh] → attention2deltax → [E'*num_head,1]
#         att_head = self.attention2deltax(
#             a.view(E * self.num_head, -1)
#         ).view(E, self.num_head)                                # [E', num_head]
#         att = self.weighted_head_layer(att_head).unsqueeze(-1)  # [E',1]

#         # 4) 残差门控
#         rg = self.residual_gate(a_flat)                         # [E',1]

#         # 5) 应用所有权重
#         dx = dx * att * rg                                      # [E',3]

#         # 6) 聚合到节点
#         agg = scatter(dx, index=ei[0], reduce='sum', dim=0,
#                       dim_size=pos.size(0))                    # [N,3]
#         out = agg[generate_node_idxes]                          # [G,3]

#         # 7) 最终归一化到模型给出的目标距离
#         d_target = generate_node_dist                           # [G,1]
#         normed = out / (
#             torch.norm(out, p=2, dim=-1, keepdim=True).clamp_min(1e-6)
#         ) * d_target                                            # [G,3]

#         return normed