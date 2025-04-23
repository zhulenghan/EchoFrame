
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
    def __init__(self, input_dim=512, hidden_dim=1024, output_dim=512, dropout=0.1):
        super().__init__()

        # self.linear = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.blocks = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            V2ABlock(hidden_dim, expansion_rate=2, dropout=0.1),
            V2ABlock(hidden_dim, expansion_rate=2, dropout=0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        pooled = x.mean(dim=1)  # (B, 512)
        return self.blocks(pooled)

class V2AMapperTr(nn.Module):
    def __init__(self, input_dim=512, model_dim=768, num_tokens=64, num_heads=12, num_layers=1, expansion_ratio=4):
        super().__init__()
        self.num_tokens = num_tokens
        self.model_dim = model_dim

        # Input embedding (Linear projection)
        self.input_proj = nn.Linear(input_dim, model_dim)

        # Learnable token [CLS]-like
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))

        # Positional Encoding (learnable or sinusoidal)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens + 1, model_dim))

        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * expansion_ratio,
            activation='gelu',
            batch_first=True  # shape: [B, S, D]
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection back to 512
        self.output_proj = nn.Linear(model_dim, input_dim)

    def forward(self, x):
        """
        x: Tensor of shape [batch_size, 64, 512]
        """
        B = x.size(0)

        # Project input to model_dim
        x = self.input_proj(x)  # [B, 64, 768]

        # Expand cls_token for each batch
        cls_token = self.cls_token.expand(B, -1, -1)  # [B, 1, 768]

        # Concatenate cls token
        x = torch.cat((cls_token, x), dim=1)  # [B, 65, 768]

        # Add positional encoding
        x = x + self.pos_embedding[:, :x.size(1), :]  # [B, 65, 768]

        # Transformer encoding
        x = self.encoder(x)  # [B, 65, 768]

        # Take cls token output and project to 512
        cls_out = x[:, 0]  # [B, 768]
        out = self.output_proj(cls_out)  # [B, 512]

        return out

class V2AMapperBiLSTM(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=1024, output_dim=512, lstm_hidden=256, lstm_layers=1, dropout=0.1):
        super().__init__()

        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )

        self.blocks = nn.Sequential(
            nn.LayerNorm(2 * lstm_hidden),  # 因为 BiLSTM -> 2*lstm_hidden
            V2ABlock(2 * lstm_hidden, expansion_rate=2, dropout=dropout),
            V2ABlock(2 * lstm_hidden, expansion_rate=2, dropout=dropout),
            nn.Linear(2 * lstm_hidden, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):  # x: (B, T, 512)
        lstm_out, (hn, cn) = self.bilstm(x)  # hn: (num_layers * 2, B, H)
        # 取最后一层的两个方向的 hidden state，拼接
        forward_last = hn[-2]     # (B, H)
        backward_last = hn[-1]    # (B, H)
        pooled = torch.cat([forward_last, backward_last], dim=1)  # (B, 2*H)
        return self.blocks(pooled)


class V2ABlocknoresidule(nn.Module):
    def __init__(self, dim, expansion_rate=2, dropout=0.1):
        super().__init__()
        hidden_dim = dim * expansion_rate
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.norm(x)
        return x
    
class V2AMapperMLPImprovednoresidule(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=1024, output_dim=512, dropout=0.1):
        super().__init__()

        # self.linear = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.blocks = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            V2ABlocknoresidule(hidden_dim, expansion_rate=2, dropout=0.1),
            V2ABlocknoresidule(hidden_dim, expansion_rate=2, dropout=0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        pooled = x.mean(dim=1)  # (B, 512)
        return self.blocks(pooled)