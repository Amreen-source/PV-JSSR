"""
Multi-resolution PV panel dataset loader.

Loads paired multi-resolution satellite imagery and segmentation masks
from the Jiangsu Province dataset (Jiang et al., 2021). Each sample
provides imagery at 0.1m, 0.3m, and 0.8m GSD with binary PV masks.
"""

import os
import glob
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class PVMultiResDataset(Dataset):
    """
    Multi-resolution PV panel segmentation dataset.

    Directory structure expected:
        root/
        ├── PV01/
        │   ├── 0.1m/images/  &  0.1m/masks/
        │   ├── 0.3m/images/  &  0.3m/masks/
        │   └── 0.8m/images/  &  0.8m/masks/
        ├── PV03/ ...
        ├── PV08/ ...
        └── splits/train.txt, val.txt, test.txt
    """

    def __init__(self, root, split='train', input_gsd=0.8,
                 input_size=128, output_size=512, augment=True):
        super().__init__()
        self.root = root
        self.split = split
        self.input_gsd = input_gsd
        self.input_size = input_size
        self.output_size = output_size
        self.augment = augment and (split == 'train')

        self.samples = self._load_split()
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def _load_split(self):
        """Load sample list from split file or auto-detect."""
        split_file = os.path.join(self.root, 'splits', f'{self.split}.txt')
        samples = []

        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append(line)
        else:
            # auto-detect from directory structure
            for region in ['PV01', 'PV03', 'PV08']:
                img_dir = os.path.join(self.root, region, '0.8m', 'images')
                if os.path.isdir(img_dir):
                    for fname in sorted(os.listdir(img_dir)):
                        if fname.lower().endswith(('.png', '.tif', '.jpg')):
                            samples.append(f"{region}/{fname}")

            # basic split: 70/15/15
            random.seed(42)
            random.shuffle(samples)
            n = len(samples)
            if self.split == 'train':
                samples = samples[:int(0.7 * n)]
            elif self.split == 'val':
                samples = samples[int(0.7 * n):int(0.85 * n)]
            else:
                samples = samples[int(0.85 * n):]

        return samples

    def _load_image(self, path, size=None):
        """Load an image file and resize."""
        img = Image.open(path).convert('RGB')
        if size is not None:
            img = img.resize((size, size), Image.BICUBIC)
        return img

    def _load_mask(self, path, size=None):
        """Load a binary mask file and resize."""
        mask = Image.open(path).convert('L')
        if size is not None:
            mask = mask.resize((size, size), Image.NEAREST)
        return mask

    def _get_paths(self, sample_id):
        """Resolve file paths for all resolutions."""
        parts = sample_id.split('/')
        region = parts[0]
        fname = parts[1] if len(parts) > 1 else parts[0]

        paths = {}
        for gsd_str in ['0.1m', '0.3m', '0.8m']:
            img_path = os.path.join(self.root, region, gsd_str, 'images', fname)
            mask_path = os.path.join(self.root, region, gsd_str, 'masks', fname)

            # try different extensions
            for ext in ['', '.png', '.tif', '.jpg']:
                if os.path.exists(img_path + ext):
                    img_path = img_path + ext
                    break
            for ext in ['', '.png', '.tif', '.jpg']:
                if os.path.exists(mask_path + ext):
                    mask_path = mask_path + ext
                    break

            paths[gsd_str] = {'image': img_path, 'mask': mask_path}

        paths['region'] = region
        return paths

    def _augment(self, lr_img, hr_img, masks):
        """Apply synchronized augmentation to all images and masks."""
        # random horizontal flip
        if random.random() > 0.5:
            lr_img = TF.hflip(lr_img)
            hr_img = TF.hflip(hr_img)
            masks = {k: TF.hflip(v) for k, v in masks.items()}

        # random vertical flip
        if random.random() > 0.5:
            lr_img = TF.vflip(lr_img)
            hr_img = TF.vflip(hr_img)
            masks = {k: TF.vflip(v) for k, v in masks.items()}

        # random 90-degree rotation
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            lr_img = TF.rotate(lr_img, angle)
            hr_img = TF.rotate(hr_img, angle)
            masks = {k: TF.rotate(v, angle) for k, v in masks.items()}

        # color jitter (images only, not masks)
        if random.random() > 0.5:
            jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            lr_img = jitter(lr_img)
            hr_img = jitter(hr_img)

        return lr_img, hr_img, masks

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id = self.samples[idx]
        paths = self._get_paths(sample_id)

        # load LR input (0.8m)
        lr_img = self._load_image(paths['0.8m']['image'], self.input_size)
        lr_mask = self._load_mask(paths['0.8m']['mask'], self.input_size)

        # load HR target (0.1m)
        hr_img = self._load_image(paths['0.1m']['image'], self.output_size)
        hr_mask = self._load_mask(paths['0.1m']['mask'], self.output_size)

        # load MR mask (0.3m)
        mr_mask = self._load_mask(paths['0.3m']['mask'], self.output_size // 2)

        # create mask dict for augmentation
        mask_dict = {
            'hr': hr_mask,
            'mr': mr_mask,
            'lr': lr_mask,
        }

        # apply augmentation
        if self.augment:
            lr_img, hr_img, mask_dict = self._augment(lr_img, hr_img, mask_dict)

        # convert to tensors
        to_tensor = T.ToTensor()

        lr_tensor = self.normalize(to_tensor(lr_img))
        hr_tensor = self.normalize(to_tensor(hr_img))

        mask_hr = (to_tensor(mask_dict['hr']) > 0.5).float()
        mask_mr = (to_tensor(mask_dict['mr']) > 0.5).float()
        mask_lr = (to_tensor(mask_dict['lr']) > 0.5).float()

        return {
            'lr_image': lr_tensor,
            'hr_image': hr_tensor,
            'mask_hr': mask_hr,
            'mask_mr': mask_mr,
            'mask_lr': mask_lr,
            'gsd': torch.tensor(self.input_gsd, dtype=torch.float32),
            'sample_id': sample_id,
            'region': paths['region'],
        }


def create_dataloaders(data_root, batch_size=2, num_workers=4, input_size=128, output_size=512):
    """Create train, validation, and test data loaders."""
    train_dataset = PVMultiResDataset(data_root, split='train', input_size=input_size,
                                      output_size=output_size, augment=True)
    val_dataset = PVMultiResDataset(data_root, split='val', input_size=input_size,
                                    output_size=output_size, augment=False)
    test_dataset = PVMultiResDataset(data_root, split='test', input_size=input_size,
                                     output_size=output_size, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
