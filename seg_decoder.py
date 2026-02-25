"""
Segmentation Decoder with SR Feature Injection.

U-Net style decoder with attention gates and SR feature injection
for precise boundary delineation. Produces segmentation masks at
three resolution levels (512², 256², 128²).

Reference: Section 3.4 of the paper (Equations 12-17).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """
    Attention gate for skip connections (Eq. 13).

    Learns to suppress irrelevant background and focus on salient
    regions, particularly PV panel boundaries.
    """

    def __init__(self, gate_channels, skip_channels, inter_channels=None):
        super().__init__()
        inter_channels = inter_channels or skip_channels // 2

        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, skip_feat, gate_feat):
        """
        Args:
            skip_feat: encoder features (B, C_skip, H, W)
            gate_feat: decoder features from deeper stage (B, C_gate, H', W')
        Returns:
            gated skip features (B, C_skip, H, W)
        """
        # upsample gate to match skip spatial size
        g = self.W_gate(gate_feat)
        if g.shape[-2:] != skip_feat.shape[-2:]:
            g = F.interpolate(g, size=skip_feat.shape[-2:], mode='bilinear', align_corners=False)

        x = self.W_skip(skip_feat)
        attention = self.psi(self.relu(g + x))
        return skip_feat * attention


class SRFeatureInjection(nn.Module):
    """
    SR Feature Injection block (Eq. 14).

    Receives texture-rich features from the SR branch and injects them
    into the segmentation decoder through a learned scaling parameter.
    """

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1)
        # learnable scaling parameter, initialized to 0 for stable training
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, seg_feat, sr_feat):
        """
        Args:
            seg_feat: segmentation decoder features (B, C, H, W)
            sr_feat: SR features transformed by CTIM (B, C, H, W)
        Returns:
            injected features (B, C, H, W)
        """
        if sr_feat.shape[-2:] != seg_feat.shape[-2:]:
            sr_feat = F.interpolate(sr_feat, size=seg_feat.shape[-2:],
                                    mode='bilinear', align_corners=False)
        if sr_feat.shape[1] != seg_feat.shape[1]:
            sr_feat = F.adaptive_avg_pool2d(sr_feat, 1).expand_as(seg_feat)  # fallback
            # this shouldn't normally happen if channels are matched

        return seg_feat + self.gamma * self.conv(sr_feat)


class DecoderBlock(nn.Module):
    """Single decoder stage: upsample + skip connection + convolutions."""

    def __init__(self, in_channels, skip_channels, out_channels, use_attention_gate=True):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.attention_gate = None
        if use_attention_gate:
            self.attention_gate = AttentionGate(in_channels, skip_channels)

        # after concatenation: in_channels + skip_channels -> out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        """
        Args:
            x: features from deeper decoder stage (B, C_in, H, W)
            skip: encoder skip features (B, C_skip, 2H, 2W)
        Returns:
            decoded features (B, C_out, 2H, 2W)
        """
        x = self.upsample(x)

        # handle size mismatch
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)

        # apply attention gate to skip connection
        if self.attention_gate is not None:
            skip = self.attention_gate(skip, x)

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SegmentationDecoder(nn.Module):
    """
    Multi-resolution segmentation decoder.

    U-Net decoder with attention gates at each skip connection,
    SR feature injection for texture-enhanced boundary precision,
    and multi-resolution mask prediction heads.
    """

    def __init__(self, encoder_channels=[64, 128, 256, 512],
                 use_attention_gates=True, use_sr_injection=True):
        super().__init__()

        # decoder stages (reverse order of encoder)
        self.decoder4 = DecoderBlock(512, 256, 256, use_attention_gates)
        self.decoder3 = DecoderBlock(256, 128, 128, use_attention_gates)
        self.decoder2 = DecoderBlock(128, 64, 64, use_attention_gates)

        # final upsample to match HR output (512x512 from 128x128 features)
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # SR Feature Injection at each decoder stage
        self.sr_injection = None
        if use_sr_injection:
            self.sr_injection = nn.ModuleDict({
                'stage3': SRFeatureInjection(256),
                'stage2': SRFeatureInjection(128),
                'stage1': SRFeatureInjection(64),
            })

        # multi-resolution mask prediction heads (Eq. 15-17)
        self.mask_head_hr = nn.Sequential(
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )  # 512x512

        self.mask_head_mr = nn.Sequential(
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )  # 256x256

        self.mask_head_lr = nn.Sequential(
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid(),
        )  # 128x128

    def forward(self, encoder_features, sr_features=None):
        """
        Args:
            encoder_features: list [F1, F2, F3, F4] from shared encoder
            sr_features: list of SR branch features for injection (optional)
        Returns:
            masks: dict with 'hr', 'mr', 'lr' segmentation masks
            decoder_features: list of intermediate features for CTIM
        """
        f1, f2, f3, f4 = encoder_features
        decoder_features = []

        # decode stage 4 -> 3
        d3 = self.decoder4(f4, f3)   # (B, 256, H/4, W/4)
        if self.sr_injection and sr_features and len(sr_features) > 0:
            d3 = self.sr_injection['stage3'](d3, sr_features[0])
        decoder_features.append(d3)

        # LR mask (128x128)
        mask_lr = self.mask_head_lr(
            F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False)
            if d3.shape[1] == 128 else
            self.mask_head_lr[0:1](d3)  # fallback: just conv + sigmoid
        )

        # decode stage 3 -> 2
        d2 = self.decoder3(d3, f2)   # (B, 128, H/2, W/2)
        if self.sr_injection and sr_features and len(sr_features) > 1:
            d2 = self.sr_injection['stage2'](d2, sr_features[1])
        decoder_features.append(d2)

        # MR mask (256x256)
        mask_mr = self.mask_head_mr(d2)

        # decode stage 2 -> 1
        d1 = self.decoder2(d2, f1)   # (B, 64, H, W)
        if self.sr_injection and sr_features and len(sr_features) > 2:
            d1 = self.sr_injection['stage1'](d1, sr_features[2])
        decoder_features.append(d1)

        # final upsample for HR mask
        d0 = self.final_up(d1)       # (B, 32, 2H, 2W)

        # HR mask (512x512)
        mask_hr = self.mask_head_hr(d0)

        masks = {
            'hr': mask_hr,
            'mr': mask_mr,
            'lr': mask_lr if mask_lr.shape[1] == 1 else self.mask_head_lr(d3),
        }

        return masks, decoder_features
