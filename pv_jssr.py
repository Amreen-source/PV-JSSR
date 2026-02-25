"""
PV-JSSR: Joint Super-Resolution and Segmentation with Cross-Task Interaction.

Main model class that integrates all components:
  - Resolution-Aware Shared Encoder
  - SR Decoder with PV-Semantic Guidance
  - Segmentation Decoder with SR Feature Injection
  - Cross-Task Interaction Module (CTIM)
  - Multi-Resolution Consistency Module (MRCM)

Reference: Section 3.1, overall architecture (Fig. 1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import SharedEncoder
from .sr_decoder import SRDecoder
from .seg_decoder import SegmentationDecoder
from .ctim import CTIM
from .mrcm import MRCM


class PVJSSR(nn.Module):
    """
    PV-JSSR: Unified framework for joint 8x super-resolution and
    semantic segmentation of photovoltaic panels.

    Given a 128x128 input at 0.8m GSD, produces:
      - 512x512 super-resolved image at 0.1m GSD
      - Multi-resolution segmentation masks (512², 256², 128²)
    """

    def __init__(self, config=None):
        super().__init__()

        # defaults if no config is provided
        if config is None:
            config = self._default_config()

        enc_cfg = config.get('encoder', {})
        sr_cfg = config.get('sr_decoder', {})
        seg_cfg = config.get('seg_decoder', {})
        ctim_cfg = config.get('ctim', {})
        mrcm_cfg = config.get('mrcm', {})

        # 1. Resolution-Aware Shared Encoder
        self.encoder = SharedEncoder(
            in_channels=3,
            channels=enc_cfg.get('channels', [64, 128, 256, 512]),
            num_heads=enc_cfg.get('num_heads', [2, 4, 8, 16]),
            window_size=enc_cfg.get('window_size', 7),
            sinusoidal_dim=config.get('gsd_embedding', {}).get('sinusoidal_dim', 256),
            embed_dim=config.get('gsd_embedding', {}).get('dim', 512),
            drop_path_rate=enc_cfg.get('drop_path_rate', 0.1),
        )

        # 2. SR Decoder
        self.sr_decoder = SRDecoder(
            base_channels=sr_cfg.get('channels', 64),
            channel_mult=sr_cfg.get('channel_mult', [1, 2, 4, 4]),
            diffusion_steps=sr_cfg.get('diffusion_steps', 1000),
            ddim_steps=sr_cfg.get('ddim_steps', 50),
        )

        # 3. Segmentation Decoder
        self.seg_decoder = SegmentationDecoder(
            encoder_channels=enc_cfg.get('channels', [64, 128, 256, 512]),
            use_attention_gates=seg_cfg.get('attention_gates', True),
            use_sr_injection=seg_cfg.get('sr_injection', True),
        )

        # 4. Cross-Task Interaction Module
        self.ctim = CTIM(
            channels_list=[256, 128, 64],
            num_heads=ctim_cfg.get('num_heads', 8),
            dropout=ctim_cfg.get('dropout', 0.1),
        )

        # 5. Multi-Resolution Consistency Module
        self.mrcm = MRCM(
            feature_channels=[256, 128, 64],
            alpha=mrcm_cfg.get('alpha', 0.5),
            beta=mrcm_cfg.get('beta', 1.0),
        )

        # training stage control
        self._ctim_enabled = False
        self._mrcm_enabled = False

    def _default_config(self):
        return {
            'encoder': {
                'channels': [64, 128, 256, 512],
                'num_heads': [2, 4, 8, 16],
                'window_size': 7,
                'drop_path_rate': 0.1,
            },
            'gsd_embedding': {'sinusoidal_dim': 256, 'dim': 512},
            'sr_decoder': {
                'channels': 64,
                'channel_mult': [1, 2, 4, 4],
                'diffusion_steps': 1000,
                'ddim_steps': 50,
            },
            'seg_decoder': {'attention_gates': True, 'sr_injection': True},
            'ctim': {'num_heads': 8, 'dropout': 0.1},
            'mrcm': {'alpha': 0.5, 'beta': 1.0},
        }

    def set_stage(self, stage):
        """
        Configure model for curriculum training stage.

        Stage 1: Encoder pretraining (no CTIM, no MRCM)
        Stage 2: Joint training with CTIM
        Stage 3: Full training with MRCM
        """
        if stage == 1:
            self._ctim_enabled = False
            self._mrcm_enabled = False
        elif stage == 2:
            self._ctim_enabled = True
            self._mrcm_enabled = False
        elif stage == 3:
            self._ctim_enabled = True
            self._mrcm_enabled = True
        else:
            raise ValueError(f"Invalid stage: {stage}. Must be 1, 2, or 3.")

    def forward(self, lr_img, gsd, hr_img=None):
        """
        Forward pass.

        Args:
            lr_img: low-resolution input (B, 3, 128, 128) at 0.8m GSD
            gsd: GSD value, scalar or (B,) tensor (e.g. 0.8)
            hr_img: high-resolution target (B, 3, 512, 512) for training
        Returns:
            dict with:
                'sr_output': super-resolved image (B, 3, 512, 512)
                'masks': dict of segmentation masks at 3 resolutions
                'sr_loss_dict': SR decoder loss components (training only)
                'mrcm_loss': MRCM loss (training only, stage 3)
        """
        # handle GSD as tensor
        if not isinstance(gsd, torch.Tensor):
            gsd = torch.tensor([gsd] * lr_img.shape[0], device=lr_img.device)
        elif gsd.dim() == 0:
            gsd = gsd.unsqueeze(0).expand(lr_img.shape[0])

        # 1. Encode
        encoder_features = self.encoder(lr_img, gsd)

        # 2. Initial segmentation pass (before CTIM)
        masks, seg_features = self.seg_decoder(encoder_features, sr_features=None)

        # 3. SR decoder forward
        sr_result = self.sr_decoder(
            lr_img, hr_img=hr_img,
            seg_mask=masks['hr'].detach() if self._ctim_enabled else None,
            encoder_features=encoder_features,
        )

        # 4. Cross-Task Interaction (if enabled)
        if self._ctim_enabled and len(seg_features) >= 3:
            # use intermediate seg features and approximate SR features
            sr_proxy = [
                F.interpolate(sr_result['sr_output'].detach(), size=f.shape[-2:],
                              mode='bilinear', align_corners=False)
                for f in seg_features
            ]
            # channel projection for SR proxy features
            sr_proxy_projected = []
            for proxy, seg_f in zip(sr_proxy, seg_features):
                if proxy.shape[1] != seg_f.shape[1]:
                    proj = nn.Conv2d(proxy.shape[1], seg_f.shape[1], 1).to(proxy.device)
                    proxy = proj(proxy)
                sr_proxy_projected.append(proxy)

            sr_updated, seg_updated = self.ctim(sr_proxy_projected, seg_features, masks['hr'])

            # re-run segmentation decoder with updated features
            masks, seg_features = self.seg_decoder(encoder_features, sr_features=seg_updated)

        # 5. MRCM loss (if enabled)
        mrcm_loss = torch.tensor(0.0, device=lr_img.device)
        mrcm_details = {}
        if self._mrcm_enabled and self.training:
            mrcm_loss, mrcm_details = self.mrcm(seg_features, masks)

        output = {
            'sr_output': sr_result['sr_output'],
            'masks': masks,
            'mrcm_loss': mrcm_loss,
            'mrcm_details': mrcm_details,
        }

        # include SR training losses if available
        if 'noise_pred' in sr_result:
            output['noise_pred'] = sr_result['noise_pred']
            output['noise_target'] = sr_result['noise_target']

        return output

    def get_param_groups(self, base_lr):
        """Get parameter groups with different learning rates."""
        encoder_params = list(self.encoder.parameters())
        sr_params = list(self.sr_decoder.parameters())
        seg_params = list(self.seg_decoder.parameters())
        ctim_params = list(self.ctim.parameters())
        mrcm_params = list(self.mrcm.parameters())

        return [
            {'params': encoder_params, 'lr': base_lr},
            {'params': sr_params, 'lr': base_lr},
            {'params': seg_params, 'lr': base_lr},
            {'params': ctim_params, 'lr': base_lr * 0.1},  # lower LR for CTIM initially
            {'params': mrcm_params, 'lr': base_lr * 0.1},
        ]
