# -*- coding: utf-8 -*-
# @Author  : hanyx9010@163.com
# @Time    : 2025/4/20 19:00
# @File    : CLloss.py
# @Description    :
import torch
import torch.nn.functional as F

def graph_contrastive_loss(z1, z2, temperature=0.5):
    """
    z1, z2: shape [batch_size, dim]，为同一批图的两个增强视图的嵌入结果
    """
    batch_size = z1.shape[0]
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)  # [2*B, D]

    sim_matrix = torch.matmul(z, z.T)  # cosine similarity
    sim_matrix = sim_matrix / temperature

    # 构造正对比样本的标签（正对在对角线下方）
    pos_mask = torch.eye(batch_size, dtype=torch.bool).to(z.device)
    pos = sim_matrix[:batch_size, batch_size:][pos_mask].unsqueeze(1)  # [B, 1]

    # 构造负样本 mask（不包括自己）
    neg_mask = ~torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
    sim_matrix = sim_matrix.masked_select(neg_mask).view(2 * batch_size, -1)

    logits = torch.cat([pos, sim_matrix[:batch_size]], dim=1)
    labels = torch.zeros(batch_size, dtype=torch.long).to(z.device)

    loss = F.cross_entropy(logits, labels)
    return loss
