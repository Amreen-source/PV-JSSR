"""
Loss functions for PV-JSSR training.

Includes SR losses (Charbonnier pixel, VGG perceptual, adversarial),
segmentation losses (BCE + Dice + boundary), CTIM alignment loss,
and PV-specific counting/area losses.

Reference: Section 3.7 (Equations 29-35) and Table 3.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CharbonnierLoss(nn.Module):
    """Charbonnier penalty for robust pixel-wise SR loss (Eq. 30)."""

    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.eps2 = epsilon ** 2

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps2))


class VGGPerceptualLoss(nn.Module):
    """VGG-19 perceptual loss (Eq. 31)."""

    def __init__(self, layers=None):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.blocks = nn.ModuleList()

        if layers is None:
            layers = [2, 7, 12, 21, 30]  # relu1_2, relu2_2, relu3_2, relu4_2, relu5_2

        prev = 0
        for idx in layers:
            self.blocks.append(vgg[prev:idx + 1])
            prev = idx + 1

        for param in self.parameters():
            param.requires_grad = False

        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std

        loss = 0.0
        x, y = pred, target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += F.l1_loss(x, y)

        return loss


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation (Eq. 34)."""

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        return 1.0 - dice


class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss that encourages precise panel edge delineation.

    Computes boundary maps from segmentation masks using Sobel-like
    edge detection and penalizes boundary misalignment.
    """

    def __init__(self):
        super().__init__()
        # Sobel kernels for boundary detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))

    def get_boundary(self, mask):
        """Extract boundary map from binary mask."""
        if mask.shape[1] > 1:
            mask = mask[:, :1]
        edge_x = F.conv2d(mask, self.sobel_x, padding=1)
        edge_y = F.conv2d(mask, self.sobel_y, padding=1)
        boundary = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)
        return boundary

    def forward(self, pred_mask, target_mask):
        pred_boundary = self.get_boundary(pred_mask)
        target_boundary = self.get_boundary(target_mask)
        return F.mse_loss(pred_boundary, target_boundary)


class CTIMLoss(nn.Module):
    """
    CTIM alignment and consistency loss (Eq. 23-25).

    Encourages meaningful cross-task feature alignment between
    the SR and segmentation branches.
    """

    def __init__(self):
        super().__init__()
        self.boundary_loss = BoundaryLoss()

    def forward(self, sr_output, mask_pred, mask_target=None):
        """
        Args:
            sr_output: super-resolved image (B, 3, H, W)
            mask_pred: predicted segmentation mask (B, 1, H, W)
            mask_target: ground truth mask (optional)
        """
        # edge-boundary consistency loss (Eq. 25)
        # detect edges in SR output
        sr_gray = sr_output.mean(dim=1, keepdim=True)
        sr_edges = self.boundary_loss.get_boundary(sr_gray)

        # get boundary from predicted mask
        mask_boundary = self.boundary_loss.get_boundary(mask_pred)

        # resize if needed
        if sr_edges.shape[-2:] != mask_boundary.shape[-2:]:
            sr_edges = F.interpolate(sr_edges, size=mask_boundary.shape[-2:],
                                     mode='bilinear', align_corners=False)

        consistency_loss = F.binary_cross_entropy_with_logits(sr_edges, mask_boundary.detach())

        return consistency_loss


class PVJSSRLoss(nn.Module):
    """
    Combined loss for PV-JSSR training (Eq. 1).

    L_total = λ_SR * L_SR + λ_Seg * L_Seg + λ_bnd * L_boundary
              + λ_1 * L_CTIM + λ_2 * L_MRCM
    """

    def __init__(self, weights=None):
        super().__init__()

        if weights is None:
            weights = {
                'sr': 1.5, 'seg': 1.5, 'boundary': 0.5,
                'ctim': 0.2, 'mrcm': 0.1,
                'pixel': 1.0, 'perceptual': 0.1, 'adversarial': 0.01,
            }
        self.weights = weights

        # SR losses
        self.pixel_loss = CharbonnierLoss()
        self.perceptual_loss = VGGPerceptualLoss()

        # Segmentation losses
        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss()
        self.boundary_loss = BoundaryLoss()

        # CTIM loss
        self.ctim_loss = CTIMLoss()

    def forward(self, outputs, targets, stage=3):
        """
        Compute total training loss.

        Args:
            outputs: model output dict
            targets: dict with 'hr_image', 'mask_hr', 'mask_mr', 'mask_lr'
            stage: curriculum training stage (1, 2, or 3)
        Returns:
            total_loss, loss_dict with individual components
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=outputs['sr_output'].device)

        # --- SR Loss ---
        sr_pred = outputs['sr_output']
        hr_target = targets['hr_image']

        # ensure same size
        if sr_pred.shape[-2:] != hr_target.shape[-2:]:
            sr_pred = F.interpolate(sr_pred, size=hr_target.shape[-2:],
                                    mode='bilinear', align_corners=False)

        l_pixel = self.pixel_loss(sr_pred, hr_target)
        l_perceptual = self.perceptual_loss(sr_pred, hr_target)

        l_sr = self.weights['pixel'] * l_pixel + self.weights['perceptual'] * l_perceptual

        # diffusion noise loss if available
        if 'noise_pred' in outputs:
            l_noise = F.mse_loss(outputs['noise_pred'], outputs['noise_target'])
            l_sr = l_sr + l_noise

        total_loss = total_loss + self.weights['sr'] * l_sr
        loss_dict['sr_pixel'] = l_pixel.item()
        loss_dict['sr_perceptual'] = l_perceptual.item()

        # --- Segmentation Loss ---
        masks = outputs['masks']
        l_seg = torch.tensor(0.0, device=total_loss.device)

        for key, target_key in [('hr', 'mask_hr'), ('mr', 'mask_mr'), ('lr', 'mask_lr')]:
            if key in masks and target_key in targets:
                pred_mask = masks[key]
                gt_mask = targets[target_key]

                if pred_mask.shape[-2:] != gt_mask.shape[-2:]:
                    gt_mask = F.interpolate(gt_mask.float(), size=pred_mask.shape[-2:],
                                            mode='nearest')

                l_bce = self.bce_loss(pred_mask, gt_mask.float())
                l_dice = self.dice_loss(pred_mask, gt_mask.float())
                l_seg = l_seg + l_bce + l_dice

        total_loss = total_loss + self.weights['seg'] * l_seg
        loss_dict['seg'] = l_seg.item()

        # --- Boundary Loss ---
        if 'mask_hr' in targets:
            gt_hr = targets['mask_hr']
            pred_hr = masks['hr']
            if pred_hr.shape[-2:] != gt_hr.shape[-2:]:
                gt_hr = F.interpolate(gt_hr.float(), size=pred_hr.shape[-2:], mode='nearest')
            l_bnd = self.boundary_loss(pred_hr, gt_hr.float())
            total_loss = total_loss + self.weights['boundary'] * l_bnd
            loss_dict['boundary'] = l_bnd.item()

        # --- CTIM Loss (stage 2+) ---
        if stage >= 2:
            l_ctim = self.ctim_loss(sr_pred, masks['hr'])
            total_loss = total_loss + self.weights['ctim'] * l_ctim
            loss_dict['ctim'] = l_ctim.item()

        # --- MRCM Loss (stage 3) ---
        if stage >= 3 and 'mrcm_loss' in outputs:
            total_loss = total_loss + self.weights['mrcm'] * outputs['mrcm_loss']
            loss_dict['mrcm'] = outputs['mrcm_loss'].item()

        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict
