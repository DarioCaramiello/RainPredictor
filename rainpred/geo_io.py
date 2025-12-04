import os
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import rasterio

from .data import normalize_image, compute_required_padding
from .config import PATCH_HEIGHT, PATCH_WIDTH

def get_inference_transform() -> transforms.Compose:
    """Return the transform used at inference time (ToTensor + Normalize)."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

def load_sequence_from_dir(
    input_dir: str,
    m: int,
    pattern: str = ".tif",
    patch_height: int = PATCH_HEIGHT,
    patch_width: int = PATCH_WIDTH,
) -> Tuple[torch.Tensor, List[str], Tuple[int, int, int, int], Dict[str, Any]]:
    """Load m GeoTIFF frames, pad to patch-aligned size, return tensor + metadata."""
    input_dir = os.path.abspath(input_dir)
    files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(pattern) or f.endswith(pattern + "f")
    ]
    files = sorted(files)
    if len(files) < m:
        raise RuntimeError(f"Not enough frames in {input_dir}: {len(files)} < {m}")
    files = files[:m]
    with rasterio.open(files[0]) as src0:
        orig_height = src0.height
        orig_width = src0.width
        meta = src0.meta.copy()
    pad_h, pad_w = compute_required_padding(orig_height, orig_width)
    transform = get_inference_transform()
    frames: List[torch.Tensor] = []
    for path in files:
        with rasterio.open(path) as src:
            img = src.read(1).astype(np.float32)
        img = normalize_image(img)
        if pad_h > 0 or pad_w > 0:
            img = np.pad(img, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0.0)
        img = Image.fromarray((img * 255.0).astype(np.uint8))
        tensor = transform(img)
        frames.append(tensor)
    seq = torch.stack(frames, dim=0).unsqueeze(0)
    shape_info = (orig_height, orig_width, pad_h, pad_w)
    return seq, files, shape_info, meta

def save_predictions_as_geotiff(
    preds: torch.Tensor,
    output_dir: str,
    shape_info: Tuple[int, int, int, int],
    meta: Dict[str, Any],
    prefix: str = "pred",
    as_dbz: bool = True,
    out_names: Optional[List[str]] = None,
) -> List[str]:
    """Save predicted frames as GeoTIFF, cropping padding and restoring metadata.

    If out_names is provided, it must contain at least as many basenames as
    there are predicted frames; these names (with extension) will be used
    instead of the generic prefix_XXX.tif pattern.
    """
    os.makedirs(output_dir, exist_ok=True)
    orig_h, orig_w, pad_h, pad_w = shape_info
    arr = preds.detach().cpu().numpy()
    arr = arr * 0.5 + 0.5
    if arr.ndim == 5:
        arr = arr[0, :, 0]
    out_meta = meta.copy()
    out_meta.update({
        "height": orig_h,
        "width": orig_w,
        "count": 1,
        "dtype": "float32",
    })
    num_frames = arr.shape[0]
    if out_names is not None and len(out_names) < num_frames:
        raise ValueError(f"out_names has len={len(out_names)} but num_frames={num_frames}")
    out_paths: List[str] = []
    for t in range(num_frames):
        frame = arr[t][:orig_h, :orig_w]
        if as_dbz:
            frame = np.clip(frame * 70.0, 0.0, 70.0).astype(np.float32)
        else:
            frame = frame.astype(np.float32)
        if out_names is not None:
            basename = out_names[t]
        else:
            basename = f"{prefix}_{t:03d}.tif"
        out_path = os.path.join(output_dir, basename)
        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(frame, 1)
        out_paths.append(out_path)
    return out_paths
