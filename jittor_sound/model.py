import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import math
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=196):  # 14x14 feature map size
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # W, W
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 相对位置编码
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

class FeatureFusionLayer(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.in_channels = in_channels
        
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )
        
        # Spatial Attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel Attention
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        channel_att = self.channel_attention(y).view(b, c, 1, 1)
        x = x * channel_att
        
        # Spatial Attention
        spatial_att = self.spatial_attention(x)
        x = x * spatial_att
        
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size=(window_size, window_size), num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(math.sqrt(L))  # 假设输入是正方形
        
        shortcut = x
        x = self.norm1(x)

        # Window Attention
        x = self.attn(x)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        
        return x

class ResNetSwinTransformer(nn.Module):
    def __init__(self, num_classes, pretrain=True):
        super().__init__()
        
        # ResNet50 特征提取器
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrain else None
        resnet = resnet50(weights=weights)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        
        # 特征维度
        self.feature_dim = 2048
        self.hidden_dim = 512
        
        # 特征融合层
        self.ffl = FeatureFusionLayer(self.feature_dim)
        
        # 特征降维
        self.projection = nn.Sequential(
            nn.Conv2d(self.feature_dim, self.hidden_dim, 1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 轻量级Swin Transformer
        self.swin_blocks = nn.ModuleList([
            SwinTransformerBlock(dim=self.hidden_dim, num_heads=8, window_size=14)  # 修改window_size为14
            for _ in range(2)
        ])
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, num_classes)
        )

    def forward(self, x):
        # ResNet特征提取 [B, 2048, 14, 14]
        features = self.feature_extractor(x)
        
        # 特征融合
        features = self.ffl(features)
        
        # 投影到较低维度 [B, 512, 14, 14]
        features = self.projection(features)
        
        # 准备给Swin Transformer的输入 [B, 196, 512]
        B, C, H, W = features.shape
        features = features.flatten(2).transpose(1, 2)
        
        # Swin Transformer处理
        for block in self.swin_blocks:
            features = block(features)
        
        # 全局池化
        features = features.mean(dim=1)
        
        # 分类
        output = self.classifier(features)
        
        return output

# 特征提取器包装类
class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        
    @torch.no_grad()
    def extract_features(self, x):
        features = self.model.feature_extractor(x)
        features = self.model.ffl(features)
        features = self.model.projection(features)
        B, C, H, W = features.shape
        features = features.flatten(2).transpose(1, 2)
        return features 