"""
PV-JSSR Training Script.

Implements the 3-stage curriculum training strategy:
  Stage 1 (epochs 1-50):   Encoder pretraining with independent decoders
  Stage 2 (epochs 51-150): Joint training with CTIM activation
  Stage 3 (epochs 151-180): Full fine-tuning with MRCM
"""

import os
import sys
import time
import argparse
import math
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models import PVJSSR
from data.pv_dataset import create_dataloaders
from losses.losses import PVJSSRLoss
from utils.metrics import compute_iou, compute_dice, compute_psnr, MetricTracker


def parse_args():
    parser = argparse.ArgumentParser(description='PV-JSSR Training')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./experiments')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_current_stage(epoch, stage1_end, stage2_end):
    """Determine curriculum training stage from epoch number."""
    if epoch <= stage1_end:
        return 1
    elif epoch <= stage2_end:
        return 2
    else:
        return 3


def get_ctim_weight(epoch, stage2_start, stage2_end, target_weight=0.2):
    """Cosine annealing for CTIM loss weight during stage 2."""
    if epoch < stage2_start:
        return 0.0
    if epoch >= stage2_end:
        return target_weight

    progress = (epoch - stage2_start) / (stage2_end - stage2_start)
    # cosine annealing from 0 to target
    return target_weight * 0.5 * (1.0 - math.cos(math.pi * progress))


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, stage, writer, cfg):
    model.train()
    tracker = MetricTracker()

    for batch_idx, batch in enumerate(loader):
        lr_img = batch['lr_image'].to(device)
        hr_img = batch['hr_image'].to(device)
        gsd = batch['gsd'].to(device)

        targets = {
            'hr_image': hr_img,
            'mask_hr': batch['mask_hr'].to(device),
            'mask_mr': batch['mask_mr'].to(device),
            'mask_lr': batch['mask_lr'].to(device),
        }

        # forward
        outputs = model(lr_img, gsd, hr_img=hr_img)

        # compute loss
        total_loss, loss_dict = criterion(outputs, targets, stage=stage)

        # backward
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        tracker.update(loss_dict)

        # logging
        global_step = epoch * len(loader) + batch_idx
        if batch_idx % cfg['training'].get('log_freq', 100) == 0:
            print(f"  Epoch {epoch} [{batch_idx}/{len(loader)}] "
                  f"Stage {stage} | Loss: {loss_dict['total']:.4f}")

            if writer:
                for k, v in loss_dict.items():
                    writer.add_scalar(f'train/{k}', v, global_step)

    return tracker.compute()


@torch.no_grad()
def validate(model, loader, criterion, device, stage):
    model.eval()
    tracker = MetricTracker()

    for batch in loader:
        lr_img = batch['lr_image'].to(device)
        hr_img = batch['hr_image'].to(device)
        gsd = batch['gsd'].to(device)

        targets = {
            'hr_image': hr_img,
            'mask_hr': batch['mask_hr'].to(device),
            'mask_mr': batch['mask_mr'].to(device),
            'mask_lr': batch['mask_lr'].to(device),
        }

        outputs = model(lr_img, gsd, hr_img=hr_img)
        _, loss_dict = criterion(outputs, targets, stage=stage)

        # compute evaluation metrics on HR mask
        mask_pred = outputs['masks']['hr']
        mask_gt = targets['mask_hr']

        iou = compute_iou(mask_pred, mask_gt)
        dice = compute_dice(mask_pred, mask_gt)
        psnr = compute_psnr(outputs['sr_output'], hr_img)

        metrics = {**loss_dict, 'iou': iou, 'dice': dice, 'psnr': psnr}
        tracker.update(metrics)

    return tracker.compute()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # setup
    torch.manual_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)

    writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))

    # data
    train_loader, val_loader, _ = create_dataloaders(
        args.data_dir,
        batch_size=cfg['training']['batch_size'],
        num_workers=cfg['data'].get('num_workers', 4),
        input_size=cfg['data']['input_size'],
        output_size=cfg['data']['output_size'],
    )
    print(f"Train: {len(train_loader.dataset)} samples | Val: {len(val_loader.dataset)} samples")

    # model
    model = PVJSSR(cfg.get('model', None)).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {total_params:.1f}M")

    # loss
    criterion = PVJSSRLoss(cfg['training'].get('loss_weights', None)).to(device)

    # optimizer
    opt_cfg = cfg['training']['optimizer']
    optimizer = optim.AdamW(
        model.parameters(),
        lr=opt_cfg['lr'],
        weight_decay=opt_cfg['weight_decay'],
        betas=tuple(opt_cfg.get('betas', [0.9, 0.999])),
    )

    # scheduler
    sched_cfg = cfg['training'].get('scheduler', {})
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg['training']['total_epochs'],
        eta_min=sched_cfg.get('eta_min', 1e-6),
    )

    # curriculum stage boundaries
    stage1_end = cfg['training']['stage1_epochs']
    stage2_end = stage1_end + cfg['training']['stage2_epochs']
    total_epochs = cfg['training']['total_epochs']

    start_epoch = 1
    best_iou = 0.0

    # resume if specified
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_iou = checkpoint.get('best_iou', 0.0)
        print(f"Resumed from epoch {start_epoch - 1}, best IoU: {best_iou:.4f}")

    # training loop
    for epoch in range(start_epoch, total_epochs + 1):
        stage = get_current_stage(epoch, stage1_end, stage2_end)

        # update model stage (enables/disables CTIM and MRCM)
        model.set_stage(stage)

        # adjust CTIM loss weight via cosine annealing
        ctim_w = get_ctim_weight(epoch, stage1_end + 1, stage2_end,
                                 cfg['training']['loss_weights'].get('ctim', 0.2))
        criterion.weights['ctim'] = ctim_w

        # reduce learning rate in stage 3
        if epoch == stage2_end + 1:
            for pg in optimizer.param_groups:
                pg['lr'] = pg['lr'] * 0.1
            print("Stage 3: Reduced learning rate by 10x")

        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, stage, writer, cfg
        )
        train_time = time.time() - t0

        scheduler.step()

        # validation
        val_freq = cfg['training'].get('val_freq', 5)
        if epoch % val_freq == 0 or epoch == total_epochs:
            val_metrics = validate(model, val_loader, criterion, device, stage)

            print(f"\nEpoch {epoch}/{total_epochs} (Stage {stage}) "
                  f"[{train_time:.0f}s] | "
                  f"Train Loss: {train_metrics.get('total', 0):.4f} | "
                  f"Val IoU: {val_metrics.get('iou', 0):.4f} | "
                  f"Val PSNR: {val_metrics.get('psnr', 0):.2f} dB")

            if writer:
                for k, v in val_metrics.items():
                    writer.add_scalar(f'val/{k}', v, epoch)

            # save best model
            current_iou = val_metrics.get('iou', 0)
            if current_iou > best_iou:
                best_iou = current_iou
                save_path = os.path.join(args.output_dir, 'checkpoints', 'pvjssr_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_iou': best_iou,
                    'config': cfg,
                }, save_path)
                print(f"  -> New best model! IoU: {best_iou:.4f}")

        # periodic checkpoint
        save_freq = cfg['training'].get('save_freq', 10)
        if epoch % save_freq == 0:
            save_path = os.path.join(args.output_dir, 'checkpoints', f'epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
                'config': cfg,
            }, save_path)

    writer.close()
    print(f"\nTraining complete. Best IoU: {best_iou:.4f}")


if __name__ == '__main__':
    main()
