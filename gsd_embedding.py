"""
Learnable GSD (Ground Sample Distance) Embedding.

Conditions the encoder on input resolution through sinusoidal positional
encoding projected via a two-layer MLP. This allows a single model to
handle variable-resolution inputs (0.1m, 0.3m, 0.8m GSD) adaptively.

Reference: Section 3.2.2 of the paper (Equations 4-6).
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal encoding of the scalar GSD value."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, gsd):
        """
        Args:
            gsd: scalar or (B,) tensor of GSD values (e.g. 0.8 for 0.8m)
        Returns:
            (B, dim) positional encoding
        """
        if not isinstance(gsd, torch.Tensor):
            gsd = torch.tensor([gsd], dtype=torch.float32)
        if gsd.dim() == 0:
            gsd = gsd.unsqueeze(0)

        device = gsd.device
        half_dim = self.dim // 2
        freq = torch.exp(
            -math.log(10000.0) * torch.arange(half_dim, device=device) / half_dim
        )

        # (B, 1) * (1, half_dim) -> (B, half_dim)
        args = gsd.unsqueeze(-1) * freq.unsqueeze(0)
        encoding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return encoding


class GSDEmbedding(nn.Module):
    """
    Resolution-aware GSD embedding module.

    Takes a scalar GSD value and produces a d-dimensional embedding vector
    that gets added to feature maps at each encoder stage to condition
    the network on input resolution.
    """

    def __init__(self, sinusoidal_dim=256, embed_dim=512):
        super().__init__()

        self.positional_encoding = SinusoidalPositionalEncoding(sinusoidal_dim)

        # two-layer MLP projection (Eq. 5)
        self.mlp = nn.Sequential(
            nn.Linear(sinusoidal_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # per-stage projection to match channel dimensions
        # stages have channels: 64, 128, 256, 512
        self.stage_projections = nn.ModuleList([
            nn.Linear(embed_dim, 64),
            nn.Linear(embed_dim, 128),
            nn.Linear(embed_dim, 256),
            nn.Linear(embed_dim, 512),
        ])

    def forward(self, gsd):
        """
        Args:
            gsd: scalar GSD value or (B,) tensor
        Returns:
            list of 4 embeddings, one per encoder stage,
            each shaped (B, C_i) where C_i is the channel dim for stage i
        """
        pe = self.positional_encoding(gsd)       # (B, sinusoidal_dim)
        embedding = self.mlp(pe)                  # (B, embed_dim)

        stage_embeddings = []
        for proj in self.stage_projections:
            e = proj(embedding)                   # (B, C_i)
            stage_embeddings.append(e)

        return stage_embeddings

    def get_conditioning(self, gsd, spatial_shapes):
        """
        Generate spatially-broadcast embeddings for adding to feature maps.

        Args:
            gsd: scalar or (B,) tensor
            spatial_shapes: list of (H, W) tuples for each stage
        Returns:
            list of tensors shaped (B, C_i, H_i, W_i)
        """
        stage_embeddings = self.forward(gsd)
        conditioned = []

        for emb, (h, w) in zip(stage_embeddings, spatial_shapes):
            # (B, C) -> (B, C, 1, 1) -> (B, C, H, W)
            spatial_emb = emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)
            conditioned.append(spatial_emb)

        return conditioned
