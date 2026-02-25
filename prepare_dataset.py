"""
Dataset Preparation Script for PV-JSSR.

Downloads and organizes the Jiangsu Province multi-resolution PV panel
dataset into the required directory structure for training.

Dataset source: Jiang et al. (2021)
  "Multi-resolution dataset for photovoltaic panel mapping from
   high-resolution satellite imagery"
  https://doi.org/10.5194/essd-13-5493-2021
"""

import os
import sys
import random
import shutil
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare PV-JSSR dataset')
    parser.add_argument('--raw_dir', type=str, required=True,
                        help='Path to downloaded raw dataset')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Output directory for organized dataset')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_ratio', type=float, default=0.70)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    return parser.parse_args()


def organize_region(raw_dir, output_dir, region):
    """Organize a single region's data into the expected structure."""
    region_dir = os.path.join(output_dir, region)

    for gsd_str in ['0.1m', '0.3m', '0.8m']:
        os.makedirs(os.path.join(region_dir, gsd_str, 'images'), exist_ok=True)
        os.makedirs(os.path.join(region_dir, gsd_str, 'masks'), exist_ok=True)

    # search for image and mask files in the raw directory
    raw_region = os.path.join(raw_dir, region)
    if not os.path.isdir(raw_region):
        print(f"  Warning: {raw_region} not found, skipping")
        return []

    sample_ids = set()
    for root, dirs, files in os.walk(raw_region):
        for f in files:
            if f.lower().endswith(('.png', '.tif', '.tiff', '.jpg')):
                sample_ids.add(os.path.splitext(f)[0])

    # copy files to organized structure
    for sample_id in sorted(sample_ids):
        for gsd_str in ['0.1m', '0.3m', '0.8m']:
            # try to locate source files (adapt pattern to your raw data layout)
            for subfolder in ['images', 'image', 'img', '']:
                for ext in ['.png', '.tif', '.tiff', '.jpg']:
                    src = os.path.join(raw_region, gsd_str, subfolder, sample_id + ext)
                    if os.path.exists(src):
                        dst = os.path.join(region_dir, gsd_str, 'images', sample_id + ext)
                        if not os.path.exists(dst):
                            shutil.copy2(src, dst)
                        break

            for subfolder in ['masks', 'mask', 'label', 'labels', '']:
                for ext in ['.png', '.tif', '.tiff']:
                    src = os.path.join(raw_region, gsd_str, subfolder, sample_id + ext)
                    if os.path.exists(src):
                        dst = os.path.join(region_dir, gsd_str, 'masks', sample_id + ext)
                        if not os.path.exists(dst):
                            shutil.copy2(src, dst)
                        break

    # return list of sample IDs found
    return [f"{region}/{sid}" for sid in sorted(sample_ids)]


def create_splits(all_samples, output_dir, train_ratio, val_ratio, seed):
    """Create train/val/test split files."""
    random.seed(seed)
    samples = all_samples.copy()
    random.shuffle(samples)

    n = len(samples)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]

    splits_dir = os.path.join(output_dir, 'splits')
    os.makedirs(splits_dir, exist_ok=True)

    for name, split_samples in [('train', train_samples), ('val', val_samples), ('test', test_samples)]:
        path = os.path.join(splits_dir, f'{name}.txt')
        with open(path, 'w') as f:
            for s in sorted(split_samples):
                f.write(s + '\n')
        print(f"  {name}: {len(split_samples)} samples -> {path}")


def main():
    args = parse_args()

    print(f"Preparing PV-JSSR dataset")
    print(f"  Raw data: {args.raw_dir}")
    print(f"  Output:   {args.output_dir}")

    all_samples = []
    for region in ['PV01', 'PV03', 'PV08']:
        print(f"\nProcessing {region}...")
        samples = organize_region(args.raw_dir, args.output_dir, region)
        all_samples.extend(samples)
        print(f"  Found {len(samples)} samples")

    print(f"\nTotal samples: {len(all_samples)}")

    if len(all_samples) > 0:
        print("\nCreating splits...")
        create_splits(all_samples, args.output_dir, args.train_ratio, args.val_ratio, args.seed)
    else:
        print("\nNo samples found. Check your raw data directory structure.")
        print("Expected: raw_dir/PV01/0.8m/images/*.png (or .tif)")

    print("\nDone!")


if __name__ == '__main__':
    main()
