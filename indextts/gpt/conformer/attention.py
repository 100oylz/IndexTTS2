"""
实现 Conformer 中的多头自注意力机制,包括相对位置编码的多头自注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, num_features, dropout_rate: float = 0.1):
        super().__init__()
        """
        这里做出了一个重要假设，即输入的特征维度 num_features 与隐藏层维度 hidden_features 以及输出特征维度 output_features 相同
        这样做的好处是简化了代码实现，减少了参数数量，同时也符合 Conformer 的设计理念

        Args:
            num_heads: 注意力头的数量
            num_features: 输入特征的维度
            dropout_rate: dropout 概率
        """
        self.num_heads = num_heads
        self.num_features = num_features
        assert num_features % num_heads == 0, "num_features must be divisible by num_heads"
        self.dim_hidden = num_features // num_heads
        # 进行了简化，没有进行隐藏层维度的调整,默认hidden_features=num_features
        # nn.Linear(num_features,num_features)->nn.Linear(num_features,hidden_features)
        self.linear_q = nn.Linear(num_features, num_features)
        self.linear_k = nn.Linear(num_features, num_features)
        self.linear_v = nn.Linear(num_features, num_features)
        # 输出线性变换,默认num_features=hidden_features=output_features
        # nn.Linear(hidden_features,num_features)->nn.Linear(hidden_features,output_features)
        self.linear_out = nn.Linear(num_features, num_features)
        self.dropout = nn.Dropout(dropout_rate)

    def forward_qkv(self, query, key, value):
        """
        计算多头注意力机制中的 Q,K,V
        基于一个假设d_k=d_v
        Args:
            query: (batch_size, d_q, num_features)
            key: (batch_size, d_k, num_features)
            value: (batch_size, d_v, num_features)
        Returns:
            query: (batch_size, num_heads, d_q, dim_hidden)
            key: (batch_size, num_heads, d_k, dim_hidden)
            value: (batch_size, num_heads, d_v, dim_hidden)
            """
        batch_size = query.size(0)
        query = self.linear_q(query).view(
            batch_size, -1, self.num_heads, self.dim_hidden).transpose(1, 2)
        key = self.linear_k(key).view(
            batch_size, -1, self.num_heads, self.dim_hidden).transpose(1, 2)
        value = self.linear_v(value).view(
            batch_size, -1, self.num_heads, self.dim_hidden).transpose(1, 2)
        return query, key, value

    def forward_attention(self, scores, value, mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool)):
        batch_size = value.size(0)
        if (mask.size(2) > 0):
            mask = mask.unsqueeze(1).eq(0)
            mask = mask[:, :, :, scores.size(-1)]
            scores = scores.masked_fill(mask, float('-inf'))
            attn = F.softmax(scores, dim=-1).masked_fill(mask, float(0.0))

        else:
            attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, value)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.num_heads*self.dim_hidden)
        return self.linear_out(output)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool), pos_emb: torch.Tensor = torch.empty(0),
                cache: torch.Tensor = torch.zeros((0, 0, 0, 0))):
        """
        Args:
            query: (batch_size, d_q, num_features)
            key: (batch_size, d_k, num_features)
            value: (batch_size, d_v, num_features)
            mask: (batch_size, d_q, d_k) or (batch_size, 1, d_k) or (1, 1, d_k)
            cache: (batch_size, num_heads, cache_len, 2*dim_hidden)cls
        Returns:
            output: (batch_size, d_q, num_features)
        """
        q, k, v = self.forward_qkv(query, key, value)

        if (cache.size(0) > 0):
            key_cache, value_cache = torch.split(
                cache, cache.size(-1) // 2, dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
            
        new_cache = torch.cat([k, v], dim=-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(self.dim_hidden)
        output = self.forward_attention(scores, v, mask)
        return output, new_cache

class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    def __init__(self, num_heads, num_features, dropout_rate: float = 0.1):
        super().__init__(num_heads, num_features, dropout_rate)
        self.linear_pos = nn.Linear(num_features, num_features)
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.num_heads, self.dim_hidden))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.num_heads, self.dim_hidden))
        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)
        
    def forward(self, query: torch.Tensor,
            key: torch.Tensor, value: torch.Tensor,
            mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
            pos_emb: torch.Tensor = torch.empty(0),
            cache: torch.Tensor = torch.zeros((0, 0, 0, 0))
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        q,k,v=self.forward_qkv(query,key,value)
        q=q.transpose(1,2)  # (batch_size, d_q, num_heads, dim_hidden)
        if (cache.size(0) > 0):
            key_cache, value_cache = torch.split(
                cache, cache.size(-1) // 2, dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        new_cache = torch.cat([k, v], dim=-1)
        
        n_batch_pos=pos_emb.size(0)
        p=self.linear_pos(pos_emb).view(
            n_batch_pos, -1, self.num_heads, self.dim_hidden).transpose(1, 2)  # (n_batch_pos, num_heads, d_pos, dim_hidden)
        
        q_with_bias_u=(q+self.pos_bias_u).transpose(1,2)  # (batch_size, num_heads, d_q, dim_hidden)
        q_with_bias_v=(q+self.pos_bias_v).transpose(1,2)  # (batch_size, num_heads, d_q, dim_hidden)
        
        matrix_ac=torch.matmul(q_with_bias_u,k.transpose(-2,-1))  # (batch_size, num_heads, d_q, d_k)
        matrix_bd=torch.matmul(q_with_bias_v,p.transpose(-2,-1))  # (batch_size, num_heads, d_q, d_k)
        
        scores = (matrix_ac + matrix_bd) / torch.sqrt(
            self.d_k) 
        
        output = self.forward_attention(scores, v, mask)
        return output, new_cache