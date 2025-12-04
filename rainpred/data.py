import os
import glob
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import rasterio
from rasterio.errors import RasterioIOError

import torchio as tio

from .config import PIN_MEMORY, PRED_LENGTH, PATCH_HEIGHT, PATCH_WIDTH
from .utils import set_seed

# Initialize RNGs
set_seed(15)

def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize radar image (in dBZ) to [0,1], clipping and removing NaN/inf."""
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    img = np.clip(img, 0, None)
    min_dbz, max_dbz = 0.0, 70.0
    img = np.clip(img, min_dbz, max_dbz)
    return (img - min_dbz) / (max_dbz - min_dbz)

def get_augmentation_transforms():
    """Return 2D augmentation transforms applied to full spatio-temporal tensor."""
    return tio.Compose([
        tio.RandomFlip(axes=(0, 1), p=0.5),
        tio.RandomAffine(scales=(0.8, 1.2), degrees=90, p=0.5),
    ])

def tiff_is_readable_quick(path: str) -> bool:
    """Try to read a single row from band 1 to quickly test GeoTIFF readability."""
    try:
        with rasterio.open(path) as src:
            _ = src.read(1, window=((0, 1), (0, src.width)))
        return True
    except Exception:
        return False

def compute_required_padding(h: int, w: int) -> Tuple[int, int]:
    """Compute bottom/right padding required to make H,W multiple of 2*patch."""
    factor_h = 2 * PATCH_HEIGHT
    factor_w = 2 * PATCH_WIDTH
    pad_h = (factor_h - (h % factor_h)) % factor_h
    pad_w = (factor_w - (w % factor_w)) % factor_w
    return pad_h, pad_w

class RadarDataset(Dataset):
    """Dataset of sequential radar GeoTIFF frames with padding, no resize."""
    def __init__(self, data_path: str, input_length: int = 18, pred_length: int = 6,
                 is_train: bool = True, generate_mask: bool = True,
                 return_paths: bool = False, min_size: int = 1024, small_debug: bool = False):
        self.input_length = input_length
        self.pred_length = pred_length
        self.seq_length = input_length + pred_length
        self.min_size = min_size
        self.is_train = is_train
        self.generate_mask = generate_mask
        self.return_paths = return_paths

        # Scan for .tif / .tiff files recursively
        self.files: List[str] = []
        self.files += glob.glob(os.path.join(data_path, "**/*.tif"), recursive=True)
        self.files += glob.glob(os.path.join(data_path, "**/*.tiff"), recursive=True)
        self.files = sorted(self.files)

        if len(self.files) == 0:
            raise RuntimeError(f"No files found in {data_path}")
        if len(self.files) < self.seq_length:
            raise RuntimeError(f"Too few files in {data_path}: {len(self.files)} < {self.seq_length}")

        # Transform: only ToTensor+Normalize (no resize)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        self.file_validity = {}
        self.valid_indices: List[int] = []
        self.total_possible_windows = max(0, len(self.files) - self.seq_length + 1)
        self.augmentation_transforms = get_augmentation_transforms() if is_train else None

        # Pre-screen windows by quick readability
        for start_idx in range(self.total_possible_windows):
            ok = True
            for i in range(self.seq_length):
                f = self.files[start_idx + i]
                if f not in self.file_validity:
                    try:
                        size = os.path.getsize(f)
                    except FileNotFoundError:
                        size = 0
                    if size < self.min_size:
                        valid = False
                    else:
                        valid = tiff_is_readable_quick(f)
                        if not valid:
                            print(f"Invalid file (read fail): {f}")
                    self.file_validity[f] = valid
                if not self.file_validity[f]:
                    ok = False
                    break
            if ok:
                self.valid_indices.append(start_idx)

        # Optionally trim dataset for debug
        if small_debug:
            self.valid_indices = self.valid_indices[: min(8, len(self.valid_indices))]

        self.total_files = len(self.files)
        self.invalid_files = sum(1 for v in self.file_validity.values() if not v)
        self.valid_windows = len(self.valid_indices)
        self.invalid_windows = self.total_possible_windows - self.valid_windows

        print(f"\nDataset stats ({'train' if is_train else 'val'}):")
        print(f"1. Total files: {self.total_files}")
        print(f"2. Invalid files: {self.invalid_files}")
        print(f"3. Total windows: {self.total_possible_windows}")
        print(f"4. Valid windows: {self.valid_windows}")
        print(f"5. Invalid windows: {self.invalid_windows}")
        print("===============================================\n")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        max_resamples = 8
        attempts = 0
        base_idx = idx
        while attempts < max_resamples:
            start = self.valid_indices[idx]
            images = []
            failed = False
            target_frame_paths = []
            # Open first frame to compute padding
            with rasterio.open(self.files[start]) as src0:
                h0, w0 = src0.height, src0.width
            pad_h, pad_w = compute_required_padding(h0, w0)
            for i in range(self.seq_length):
                f = self.files[start + i]
                try:
                    with rasterio.open(f) as src:
                        img = src.read(1).astype(np.float32)
                except Exception:
                    self.file_validity[f] = False
                    failed = True
                    if self.is_train:
                        idx = (idx + np.random.randint(1, 64)) % len(self.valid_indices)
                    else:
                        idx = (idx + 1) % len(self.valid_indices)
                    attempts += 1
                    break
                img = normalize_image(img)
                if pad_h > 0 or pad_w > 0:
                    img = np.pad(img, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0.0)
                img = Image.fromarray((img * 255.0).astype(np.uint8))
                img = self.transform(img)
                images.append(img)
                if i >= self.input_length:
                    target_frame_paths.append(f)
            if failed:
                continue
            all_frames = torch.stack(images)       # (seq, C, H, W)
            all_frames = all_frames.permute(1, 0, 2, 3)  # (C, seq, H, W)
            if self.is_train and self.augmentation_transforms is not None:
                all_frames = self.augmentation_transforms(all_frames)
            if not isinstance(all_frames, torch.Tensor):
                all_frames = torch.as_tensor(all_frames)
            inputs = all_frames[:, : self.input_length]
            targets = all_frames[:, self.input_length :]
            inputs = inputs.permute(1, 0, 2, 3)
            targets = targets.permute(1, 0, 2, 3)
            mask = None
            if self.generate_mask:
                mask = torch.where(targets > -1.0, 1.0, 0.0)
            if self.return_paths:
                if len(target_frame_paths) != self.pred_length:
                    raise RuntimeError(
                        f"target_frame_paths len={len(target_frame_paths)} != pred_length={self.pred_length} for start={start}"
                    )
                return inputs, targets, mask, target_frame_paths
            return inputs, targets, mask
        raise RasterioIOError(f"No valid window after {max_resamples} attempts; base_idx={base_idx}.")        

def collate_val(batch):
    """Collate function for val loader that preserves paths and adds batch dim."""
    item = batch[0]
    if len(item) != 4:
        raise RuntimeError("Validation loader must return (inputs, targets, mask, paths).")
    inputs, targets, mask, paths = item
    if isinstance(paths, (list, tuple)) and paths and isinstance(paths[0], (list, tuple)):
        paths = list(paths[0])
    elif isinstance(paths, (list, tuple)):
        paths = list(paths)
    else:
        paths = [paths]
    if inputs.dim() == 4:
        inputs = inputs.unsqueeze(0)
    if targets.dim() == 4:
        targets = targets.unsqueeze(0)
    if mask is not None and mask.dim() == 4:
        mask = mask.unsqueeze(0)
    return inputs, targets, mask, paths

def create_dataloaders(data_path: str, batch_size: int, num_workers: int,
                       pred_length: int, small_debug: bool = False):
    """Create train and val DataLoaders for the padded GeoTIFF dataset."""
    train_dataset = RadarDataset(os.path.join(data_path, 'train'),
                                 is_train=True, pred_length=pred_length,
                                 small_debug=small_debug)
    val_dataset = RadarDataset(os.path.join(data_path, 'val'),
                               is_train=False, return_paths=True,
                               pred_length=pred_length,
                               small_debug=small_debug)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=collate_val,
    )
    return train_loader, val_loader
