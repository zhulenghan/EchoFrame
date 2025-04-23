
import torch
import os
from tqdm import tqdm, trange
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from vgg_dataset import *

class V2AMapper(nn.Module):
    def __init__(self, input_dim, output_dim, expansion_rate=4):
        super(V2AMapper, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim * expansion_rate)
        self.silu = nn.SiLU()
        self.layer_norm = nn.LayerNorm(input_dim * expansion_rate)
        self.linear2 = nn.Linear(input_dim * expansion_rate, output_dim)
    
    def forward(self, x):
        identity = x
        # print(x.shape)
        x = self.linear(x)
        x = self.silu(x)
        x = self.layer_norm(x)    
        x = self.linear2(x)
        # print(x.shape)
        x += identity
        
        return x



class V2AMapperMLP(nn.Module):
    """
    将(64,512)的clip特征先池化到(1,512),
    再映射到(1,512).
    """
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=512):
        super().__init__()
        # 可以先做一个简单的线性层, 或者堆叠多层
        self.pooling = nn.AdaptiveAvgPool2d((1, input_dim))  
        # pooling后, shape变成 (1, input_dim)

        self.v2a = nn.ModuleList([
            V2AMapper(input_dim, hidden_dim, expansion_rate=1),
            V2AMapper(hidden_dim, hidden_dim, expansion_rate=1),
            V2AMapper(hidden_dim, hidden_dim, expansion_rate=1),
            V2AMapper(hidden_dim, hidden_dim, expansion_rate=1),
            V2AMapper(hidden_dim, output_dim, expansion_rate=1)
        ])
    def forward(self, x):
        # x: (batch_size, 64, 512)
        # 先把 shape (B,64,512) pooling 到 (B,1,512)
        # 这里可以用简单的mean替代，也可以用AdaptiveAvgPool2d
        pooled = x.mean(dim=1)  # (B,512)

        # 送入多层感知机映射到(512)
        # out = self.v2a(pooled)  # (B,512)
        for layer in self.v2a:
            pooled = layer(pooled)
        out = pooled
        return out


class V2ABlock(nn.Module):
    def __init__(self, dim, expansion_rate=2, dropout=0.1):
        super().__init__()
        hidden_dim = dim * expansion_rate
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        identity = x
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.norm(x + identity)
        return x

class V2AMapperMLPImproved(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=1024, output_dim=512):
        super().__init__()

        # self.linear = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.blocks = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            V2ABlock(hidden_dim, expansion_rate=2),
            V2ABlock(hidden_dim, expansion_rate=2),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        pooled = x.mean(dim=1)  # (B, 512)
        return self.blocks(pooled)
