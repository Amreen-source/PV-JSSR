"""
Evaluation metrics for PV-JSSR.

Computes IoU, Dice coefficient, PSNR, and SSIM for evaluating
segmentation and super-resolution quality.
"""

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as compare_ssim


def compute_iou(pred, target, threshold=0.5):
    """Intersection over Union for binary segmentation."""
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()

    intersection = (pred_bin * target_bin).sum()
    union = pred_bin.sum() + target_bin.sum() - intersection

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return (intersection / union).item()


def compute_dice(pred, target, threshold=0.5, smooth=1e-6):
    """Dice coefficient for binary segmentation."""
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()

    intersection = (pred_bin * target_bin).sum()
    total = pred_bin.sum() + target_bin.sum()

    dice = (2.0 * intersection + smooth) / (total + smooth)
    return dice.item()


def compute_psnr(pred, target, max_val=1.0):
    """Peak Signal-to-Noise Ratio between two images."""
    mse = F.mse_loss(pred, target).item()
    if mse < 1e-10:
        return 100.0
    psnr = 10.0 * np.log10(max_val ** 2 / mse)
    return psnr


def compute_ssim(pred, target):
    """Structural Similarity Index Measure."""
    # move to numpy
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # handle batch dimension
    if pred.ndim == 4:
        ssim_vals = []
        for i in range(pred.shape[0]):
            p = np.transpose(pred[i], (1, 2, 0))
            t = np.transpose(target[i], (1, 2, 0))
            ssim_val = compare_ssim(p, t, channel_axis=2, data_range=1.0)
            ssim_vals.append(ssim_val)
        return np.mean(ssim_vals)
    elif pred.ndim == 3:
        p = np.transpose(pred, (1, 2, 0))
        t = np.transpose(target, (1, 2, 0))
        return compare_ssim(p, t, channel_axis=2, data_range=1.0)
    else:
        return compare_ssim(pred, t, data_range=1.0)


class MetricTracker:
    """Tracks and averages metrics over an evaluation epoch."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {}
        self.counts = {}

    def update(self, metric_dict, n=1):
        for key, value in metric_dict.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0
            self.metrics[key] += value * n
            self.counts[key] += n

    def compute(self):
        result = {}
        for key in self.metrics:
            if self.counts[key] > 0:
                result[key] = self.metrics[key] / self.counts[key]
            else:
                result[key] = 0.0
        return result

    def __str__(self):
        computed = self.compute()
        parts = [f"{k}: {v:.4f}" for k, v in computed.items()]
        return " | ".join(parts)
