"""
Resolution-Aware Shared Encoder.

Hierarchical multi-scale encoder combining CNN stem with Swin Transformer
blocks. Feature extraction is conditioned on input GSD through the
GSD embedding module.

Reference: Section 3.2 of the paper (Equations 2-3, 6).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

from .gsd_embedding import GSDEmbedding


class ResidualBlock(nn.Module):
    """Standard residual block with two 3x3 convolutions."""

    def __init__(self, channels, drop_path=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.drop_path(out) + residual
        return self.relu(out)


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention (W-MSA)."""

    def __init__(self, dim, num_heads, window_size=7, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer block with window attention and MLP."""

    def __init__(self, dim, num_heads, window_size=7, mlp_ratio=4.0,
                 drop=0.0, attn_drop=0.0, drop_path=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, num_heads=num_heads, window_size=window_size,
            attn_drop=attn_drop, proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # pad feature maps to be divisible by window_size
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        _, _, Hp, Wp = x.shape
        x = x.permute(0, 2, 3, 1)  # B, H, W, C

        shortcut = x

        # window attention
        x = self.norm1(x)
        x_windows = window_partition(x, self.window_size)
        attn_windows = self.attn(x_windows)
        x = window_reverse(attn_windows, self.window_size, Hp, Wp)
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x.permute(0, 3, 1, 2)  # B, C, H, W

        # remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]

        return x


class EncoderStage(nn.Module):
    """Single encoder stage: ResBlocks + Swin Transformer + optional downsampling."""

    def __init__(self, in_channels, out_channels, num_heads, num_res_blocks=2,
                 window_size=7, downsample=True, drop_path=0.0):
        super().__init__()

        # channel projection if needed
        self.proj = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        # residual blocks
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(out_channels, drop_path=drop_path)
            for _ in range(num_res_blocks)
        ])

        # transformer block
        self.transformer = SwinTransformerBlock(
            dim=out_channels, num_heads=num_heads,
            window_size=window_size, drop_path=drop_path,
        )

        # downsampling via strided conv
        self.downsample = None
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        x = self.proj(x)
        x = self.res_blocks(x)
        x = self.transformer(x)
        feat = x  # store before downsampling for skip connections

        if self.downsample is not None:
            x = self.downsample(x)

        return x, feat


class SharedEncoder(nn.Module):
    """
    Resolution-Aware Shared Encoder.

    Hierarchical encoder with 4 stages producing multi-scale features.
    Each stage consists of ResBlocks + Swin Transformer blocks.
    GSD embedding is added to features at each stage for resolution awareness.
    """

    def __init__(self, in_channels=3, channels=[64, 128, 256, 512],
                 num_heads=[2, 4, 8, 16], window_size=7,
                 sinusoidal_dim=256, embed_dim=512, drop_path_rate=0.1):
        super().__init__()

        # stem convolution (Eq. 2)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[0], channels[0], 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        # GSD embedding
        self.gsd_embedding = GSDEmbedding(sinusoidal_dim, embed_dim)

        # encoder stages
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 4)]

        self.stage1 = EncoderStage(channels[0], channels[0], num_heads[0],
                                   window_size=window_size, downsample=True, drop_path=dpr[0])
        self.stage2 = EncoderStage(channels[0], channels[1], num_heads[1],
                                   window_size=window_size, downsample=True, drop_path=dpr[1])
        self.stage3 = EncoderStage(channels[1], channels[2], num_heads[2],
                                   window_size=window_size, downsample=True, drop_path=dpr[2])
        self.stage4 = EncoderStage(channels[2], channels[3], num_heads[3],
                                   window_size=window_size, downsample=False, drop_path=dpr[3])

    def forward(self, x, gsd):
        """
        Args:
            x: input image (B, 3, H, W)
            gsd: GSD value, scalar or (B,) tensor
        Returns:
            features: list of [F1, F2, F3, F4] multi-scale features
        """
        # stem
        x = self.stem(x)

        # get GSD conditioning for each stage
        # we need to know spatial shapes, which we can compute from input
        B, _, H, W = x.shape
        spatial_shapes = [
            (H, W),           # stage 1 feature size
            (H // 2, W // 2),
            (H // 4, W // 4),
            (H // 8, W // 8),
        ]
        gsd_cond = self.gsd_embedding.get_conditioning(gsd, spatial_shapes)

        features = []

        # stage 1
        x, f1 = self.stage1(x)
        f1 = f1 + gsd_cond[0]  # add GSD embedding (Eq. 6)
        features.append(f1)

        # stage 2
        x, f2 = self.stage2(x)
        f2 = f2 + gsd_cond[1]
        features.append(f2)

        # stage 3
        x, f3 = self.stage3(x)
        f3 = f3 + gsd_cond[2]
        features.append(f3)

        # stage 4 (no downsampling)
        _, f4 = self.stage4(x)
        f4 = f4 + gsd_cond[3]
        features.append(f4)

        return features
