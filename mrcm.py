"""
Multi-Resolution Consistency Module (MRCM).

Enforces feature and prediction consistency across the three resolution
levels (0.1m, 0.3m, 0.8m GSD) available in the training data.

Reference: Section 3.6 of the paper (Equations 26-28).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureProjection(nn.Module):
    """Projects features from one scale to match another after upsampling."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.proj(x)


class MRCM(nn.Module):
    """
    Multi-Resolution Consistency Module.

    Two components:
    1. Cross-Scale Feature Alignment (Eq. 26): ensures feature representations
       remain coherent when upsampled from lower to higher resolutions
    2. Multi-Resolution Prediction Consistency (Eq. 27): enforces that
       segmentation predictions are consistent across mask resolutions
    """

    def __init__(self, feature_channels=[256, 128, 64], alpha=0.5, beta=1.0):
        super().__init__()
        self.alpha = alpha  # feature consistency weight
        self.beta = beta    # prediction consistency weight

        # feature projection layers for scale alignment
        # low -> mid resolution features
        self.proj_low_to_mid = FeatureProjection(feature_channels[0], feature_channels[1])
        # mid -> high resolution features
        self.proj_mid_to_high = FeatureProjection(feature_channels[1], feature_channels[2])

    def feature_consistency_loss(self, features_low, features_mid, features_high):
        """
        Cross-scale feature alignment loss (Eq. 26).

        Aligns upsampled lower-resolution features with their higher-resolution
        counterparts via MSE loss.
        """
        # upsample low -> mid scale and compute alignment loss
        low_up = F.interpolate(features_low, size=features_mid.shape[-2:],
                               mode='bilinear', align_corners=False)
        low_projected = self.proj_low_to_mid(low_up)
        loss_low_mid = F.mse_loss(low_projected, features_mid.detach())

        # upsample mid -> high scale
        mid_up = F.interpolate(features_mid, size=features_high.shape[-2:],
                               mode='bilinear', align_corners=False)
        mid_projected = self.proj_mid_to_high(mid_up)
        loss_mid_high = F.mse_loss(mid_projected, features_high.detach())

        return loss_low_mid + loss_mid_high

    def prediction_consistency_loss(self, mask_hr, mask_mr, mask_lr):
        """
        Multi-resolution prediction consistency loss (Eq. 27).

        Enforces that segmentation masks at different resolutions
        produce consistent predictions when spatially aligned.
        """
        # upsample MR mask to HR resolution and compute BCE
        mr_up = F.interpolate(mask_mr, size=mask_hr.shape[-2:],
                              mode='bilinear', align_corners=False)
        loss_mr_hr = F.binary_cross_entropy(mr_up, mask_hr.detach())

        # upsample LR mask to MR resolution
        lr_up = F.interpolate(mask_lr, size=mask_mr.shape[-2:],
                              mode='bilinear', align_corners=False)
        loss_lr_mr = F.binary_cross_entropy(lr_up, mask_mr.detach())

        return loss_mr_hr + loss_lr_mr

    def forward(self, decoder_features, masks):
        """
        Compute total MRCM loss (Eq. 28).

        Args:
            decoder_features: list of decoder features at 3 scales
                [feat_low (256ch), feat_mid (128ch), feat_high (64ch)]
            masks: dict with 'hr', 'mr', 'lr' segmentation masks
        Returns:
            total MRCM loss = alpha * L_feat + beta * L_pred
        """
        # feature consistency
        if len(decoder_features) >= 3:
            l_feat = self.feature_consistency_loss(
                decoder_features[0], decoder_features[1], decoder_features[2]
            )
        else:
            l_feat = torch.tensor(0.0, device=masks['hr'].device)

        # prediction consistency
        l_pred = self.prediction_consistency_loss(
            masks['hr'], masks['mr'], masks['lr']
        )

        total = self.alpha * l_feat + self.beta * l_pred
        return total, {'feat_loss': l_feat.item(), 'pred_loss': l_pred.item()}
