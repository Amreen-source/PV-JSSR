"""Visualization utilities for PV-JSSR results."""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1)


def save_qualitative_results(lr_img, sr_img, pred_mask, gt_mask, output_dir, name):
    """Save a 6-panel qualitative result figure (matching Fig. 9 in the paper)."""
    os.makedirs(output_dir, exist_ok=True)

    lr_np = denormalize(lr_img).permute(1, 2, 0).numpy()
    sr_np = denormalize(sr_img).permute(1, 2, 0).numpy()
    pred_np = pred_mask.squeeze().cpu().numpy()
    gt_np = gt_mask.squeeze().cpu().numpy()

    pred_bin = (pred_np > 0.5).astype(np.float32)
    gt_bin = (gt_np > 0.5).astype(np.float32)

    # overlay: green channel for detected panels
    overlay = sr_np.copy()
    mask_color = np.zeros_like(overlay)
    mask_color[:, :, 1] = pred_bin
    overlay = overlay * 0.7 + mask_color * 0.3

    # error map
    error = np.abs(pred_bin - gt_bin)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(lr_np)
    axes[0, 0].set_title('(a) LR Input (0.8m GSD)')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(sr_np)
    axes[0, 1].set_title('(b) HR Output (0.1m GSD)')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(np.clip(overlay, 0, 1))
    axes[0, 2].set_title('(c) SR Segmentation Overlay')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(gt_bin, cmap='gray')
    axes[1, 0].set_title('(d) Ground Truth Mask')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(pred_bin, cmap='gray')
    axes[1, 1].set_title('(e) Predicted Mask')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(error, cmap='hot')
    axes[1, 2].set_title('(f) Error Analysis')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{name}.png'), dpi=150, bbox_inches='tight')
    plt.close()
