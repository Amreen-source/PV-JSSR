"""
PV-JSSR Inference Script.

Run inference on single images or batches of satellite imagery.
Produces super-resolved images and segmentation masks.
"""

import os
import argparse
import glob
import yaml

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

from models import PVJSSR


def parse_args():
    parser = argparse.ArgumentParser(description='PV-JSSR Inference')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--input', type=str, help='Path to single image')
    parser.add_argument('--input_dir', type=str, help='Path to directory of images')
    parser.add_argument('--output', type=str, default='./results')
    parser.add_argument('--gsd', type=float, default=0.8, help='Input GSD in meters')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.5, help='Segmentation threshold')
    return parser.parse_args()


def load_image(path, size=128):
    """Load and preprocess a satellite image."""
    img = Image.open(path).convert('RGB')
    img = img.resize((size, size), Image.BICUBIC)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)


def denormalize(tensor):
    """Reverse ImageNet normalization for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor.cpu() * std + mean
    return tensor.clamp(0, 1)


def save_outputs(sr_image, mask, output_dir, filename):
    """Save super-resolved image and segmentation mask."""
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(filename)[0]

    # save SR image
    sr_np = denormalize(sr_image).squeeze(0).permute(1, 2, 0).numpy()
    sr_pil = Image.fromarray((sr_np * 255).astype(np.uint8))
    sr_pil.save(os.path.join(output_dir, f'{base}_sr.png'))

    # save segmentation mask
    mask_np = mask.squeeze().cpu().numpy()
    mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
    mask_pil.save(os.path.join(output_dir, f'{base}_mask.png'))

    # save overlay
    overlay = sr_np.copy()
    mask_rgb = np.stack([np.zeros_like(mask_np), mask_np, np.zeros_like(mask_np)], axis=-1)
    overlay = overlay * 0.7 + mask_rgb * 0.3
    overlay = np.clip(overlay, 0, 1)
    overlay_pil = Image.fromarray((overlay * 255).astype(np.uint8))
    overlay_pil.save(os.path.join(output_dir, f'{base}_overlay.png'))

    print(f"  Saved: {base}_sr.png, {base}_mask.png, {base}_overlay.png")


@torch.no_grad()
def run_inference(model, image_paths, gsd, device, output_dir, threshold=0.5):
    model.eval()

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        print(f"Processing: {filename}")

        img_tensor = load_image(img_path).to(device)
        gsd_tensor = torch.tensor([gsd], device=device)

        outputs = model(img_tensor, gsd_tensor)

        sr_output = outputs['sr_output']
        mask = (outputs['masks']['hr'] > threshold).float()

        save_outputs(sr_output, mask, output_dir, filename)

    print(f"\nDone! Results saved to {output_dir}")


def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_cfg = checkpoint.get('config', {}).get('model', None)
    model = PVJSSR(model_cfg).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.set_stage(3)
    print(f"Model loaded from {args.checkpoint}")

    # collect input images
    image_paths = []
    if args.input:
        image_paths = [args.input]
    elif args.input_dir:
        for ext in ['*.png', '*.tif', '*.tiff', '*.jpg', '*.jpeg']:
            image_paths.extend(glob.glob(os.path.join(args.input_dir, ext)))
        image_paths.sort()
    else:
        print("Error: Provide --input or --input_dir")
        return

    print(f"Found {len(image_paths)} image(s) to process")
    run_inference(model, image_paths, args.gsd, device, args.output, args.threshold)


if __name__ == '__main__':
    main()
