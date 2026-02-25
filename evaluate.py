"""
PV-JSSR Evaluation Script.

Evaluates a trained model on the test set, computing IoU, Dice, PSNR,
and SSIM metrics. Supports per-region breakdown and visualization output.
"""

import os
import argparse
import yaml
import numpy as np
from collections import defaultdict

import torch
from tqdm import tqdm

from models import PVJSSR
from data.pv_dataset import create_dataloaders
from utils.metrics import compute_iou, compute_dice, compute_psnr, compute_ssim, MetricTracker
from utils.visualization import save_qualitative_results


def parse_args():
    parser = argparse.ArgumentParser(description='PV-JSSR Evaluation')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_vis', action='store_true', help='Save qualitative results')
    parser.add_argument('--per_region', action='store_true', help='Report per-region metrics')
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, test_loader, device, save_vis=False, output_dir=None, per_region=False):
    model.eval()

    overall_tracker = MetricTracker()
    region_trackers = defaultdict(MetricTracker)

    for batch_idx, batch in enumerate(tqdm(test_loader, desc='Evaluating')):
        lr_img = batch['lr_image'].to(device)
        hr_img = batch['hr_image'].to(device)
        gsd = batch['gsd'].to(device)

        # forward pass (inference mode)
        outputs = model(lr_img, gsd)

        sr_output = outputs['sr_output']
        mask_pred = outputs['masks']['hr']
        mask_gt = batch['mask_hr'].to(device)

        # compute metrics
        iou = compute_iou(mask_pred, mask_gt)
        dice = compute_dice(mask_pred, mask_gt)
        psnr = compute_psnr(sr_output, hr_img)
        ssim = compute_ssim(sr_output, hr_img)

        metrics = {'iou': iou, 'dice': dice, 'psnr': psnr, 'ssim': ssim}
        overall_tracker.update(metrics)

        # per-region tracking
        if per_region:
            region = batch['region'][0]
            region_trackers[region].update(metrics)

        # save visualizations
        if save_vis and output_dir and batch_idx < 50:
            save_qualitative_results(
                lr_img[0], sr_output[0], mask_pred[0], mask_gt[0],
                os.path.join(output_dir, 'visualizations'),
                batch['sample_id'][0].replace('/', '_'),
            )

    # print results
    overall = overall_tracker.compute()
    print("\n" + "=" * 60)
    print("Overall Test Results:")
    print(f"  IoU:  {overall['iou']:.4f}")
    print(f"  Dice: {overall['dice']:.4f}")
    print(f"  PSNR: {overall['psnr']:.2f} dB")
    print(f"  SSIM: {overall['ssim']:.4f}")

    if per_region:
        print("\nPer-Region Results:")
        for region in sorted(region_trackers.keys()):
            r = region_trackers[region].compute()
            print(f"  {region}: IoU={r['iou']:.4f} | Dice={r['dice']:.4f} | "
                  f"PSNR={r['psnr']:.2f} | SSIM={r['ssim']:.4f}")

    print("=" * 60)
    return overall


def main():
    args = parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_cfg = checkpoint.get('config', cfg).get('model', None)
    model = PVJSSR(model_cfg).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.set_stage(3)  # full model for evaluation
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

    # data
    _, _, test_loader = create_dataloaders(
        args.data_dir,
        batch_size=1,
        num_workers=cfg['data'].get('num_workers', 4),
        input_size=cfg['data']['input_size'],
        output_size=cfg['data']['output_size'],
    )
    print(f"Test samples: {len(test_loader.dataset)}")

    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_vis:
        os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)

    evaluate(model, test_loader, device,
             save_vis=args.save_vis, output_dir=args.output_dir,
             per_region=args.per_region)


if __name__ == '__main__':
    main()
