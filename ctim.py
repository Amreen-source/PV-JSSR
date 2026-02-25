"""
Cross-Task Interaction Module (CTIM).

Enables bidirectional feature exchange between SR and segmentation branches:
  - SR -> Seg: boundary-aware attention transfers texture details for
    sharper segmentation boundaries
  - Seg -> SR: adaptive masking focuses SR reconstruction on PV panel regions

Reference: Section 3.5 of the paper (Equations 18-25).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """Multi-head cross-attention between two feature sets."""

    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query_feat, kv_feat):
        """
        Args:
            query_feat: (B, N, C) queries from one branch
            kv_feat: (B, M, C) keys/values from the other branch
        Returns:
            (B, N, C) cross-attended features
        """
        B, N, C = query_feat.shape
        M = kv_feat.shape[1]

        q = self.q_proj(query_feat).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(kv_feat).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(kv_feat).reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        out = self.proj_drop(out)

        return out


class BoundaryAwareConv(nn.Module):
    """3x3 conv with boundary-aware features for the SR->Seg path."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        # edge detection kernel (Sobel-like, learnable init)
        self.edge_conv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)

    def forward(self, x):
        edge_feat = self.edge_conv(x)
        return self.conv(x + edge_feat)


class GatedFusion(nn.Module):
    """
    Gated fusion of task-specific and cross-task features (Eq. 20-22).

    Learns a gate G in [0,1] to balance original features with
    cross-task features adaptively.
    """

    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, task_feat, cross_feat):
        """
        Args:
            task_feat: original task features (B, C, H, W)
            cross_feat: features from the other branch (B, C, H, W)
        Returns:
            fused features (B, C, H, W)
        """
        g = self.gate(torch.cat([task_feat, cross_feat], dim=1))
        fused = g * task_feat + (1 - g) * cross_feat
        return fused


class CTIMBlock(nn.Module):
    """
    Single CTIM block operating at one scale level.

    Performs bidirectional cross-attention between SR and Seg features,
    then applies gated fusion to combine original and cross-task info.
    """

    def __init__(self, channels, num_heads=8, dropout=0.1):
        super().__init__()

        # SR -> Seg: boundary-aware attention (Eq. 18)
        self.sr_to_seg_attn = CrossAttention(channels, num_heads, attn_drop=dropout, proj_drop=dropout)
        self.boundary_conv = BoundaryAwareConv(channels)

        # Seg -> SR: adaptive masking attention (Eq. 19)
        self.seg_to_sr_attn = CrossAttention(channels, num_heads, attn_drop=dropout, proj_drop=dropout)

        # gated fusion for each branch
        self.sr_gate = GatedFusion(channels)
        self.seg_gate = GatedFusion(channels)

        # post-processing
        self.sr_norm = nn.LayerNorm(channels)
        self.seg_norm = nn.LayerNorm(channels)

        self.sr_ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout),
        )
        self.seg_ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout),
        )

    def forward(self, sr_feat, seg_feat, seg_mask=None):
        """
        Args:
            sr_feat: SR branch features (B, C, H, W)
            seg_feat: Seg branch features (B, C, H, W)
            seg_mask: intermediate segmentation prediction (B, 1, H, W) or None
        Returns:
            sr_out: updated SR features
            seg_out: updated Seg features
        """
        B, C, H, W = sr_feat.shape

        # flatten spatial dims for attention
        sr_flat = sr_feat.flatten(2).transpose(1, 2)    # (B, HW, C)
        seg_flat = seg_feat.flatten(2).transpose(1, 2)  # (B, HW, C)

        # SR -> Seg direction (Eq. 18): seg queries, sr keys/values
        sr_to_seg = self.sr_to_seg_attn(seg_flat, sr_flat)
        sr_to_seg = sr_to_seg.transpose(1, 2).reshape(B, C, H, W)
        sr_to_seg = self.boundary_conv(sr_to_seg)

        # Seg -> SR direction (Eq. 19): sr queries, seg keys/values
        seg_to_sr = self.seg_to_sr_attn(sr_flat, seg_flat)
        seg_to_sr = seg_to_sr.transpose(1, 2).reshape(B, C, H, W)

        # apply adaptive masking if segmentation mask is available
        if seg_mask is not None:
            mask_resized = F.interpolate(seg_mask, size=(H, W), mode='bilinear', align_corners=False)
            mask_weight = torch.sigmoid(mask_resized)
            seg_to_sr = seg_to_sr * mask_weight

        # gated fusion (Eq. 20-22)
        sr_fused = self.sr_gate(sr_feat, seg_to_sr)
        seg_fused = self.seg_gate(seg_feat, sr_to_seg)

        # FFN refinement with residual
        sr_out = sr_fused.flatten(2).transpose(1, 2)
        sr_out = sr_out + self.sr_ffn(self.sr_norm(sr_out))
        sr_out = sr_out.transpose(1, 2).reshape(B, C, H, W)

        seg_out = seg_fused.flatten(2).transpose(1, 2)
        seg_out = seg_out + self.seg_ffn(self.seg_norm(seg_out))
        seg_out = seg_out.transpose(1, 2).reshape(B, C, H, W)

        return sr_out, seg_out


class CTIM(nn.Module):
    """
    Cross-Task Interaction Module.

    Operates at multiple scales, enabling bidirectional feature exchange
    between the SR and segmentation decoder branches.
    """

    def __init__(self, channels_list=[256, 128, 64], num_heads=8, dropout=0.1):
        super().__init__()

        self.blocks = nn.ModuleList([
            CTIMBlock(ch, num_heads=num_heads, dropout=dropout)
            for ch in channels_list
        ])

    def forward(self, sr_features, seg_features, seg_mask=None):
        """
        Args:
            sr_features: list of SR decoder features at different scales
            seg_features: list of Seg decoder features at different scales
            seg_mask: intermediate segmentation prediction
        Returns:
            sr_updated: list of updated SR features
            seg_updated: list of updated Seg features
        """
        sr_updated = []
        seg_updated = []

        for i, block in enumerate(self.blocks):
            sr_f = sr_features[i] if i < len(sr_features) else sr_features[-1]
            seg_f = seg_features[i] if i < len(seg_features) else seg_features[-1]

            sr_out, seg_out = block(sr_f, seg_f, seg_mask)
            sr_updated.append(sr_out)
            seg_updated.append(seg_out)

        return sr_updated, seg_updated
