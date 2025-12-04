from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import confusion_matrix

criterion_sl1 = nn.SmoothL1Loss()
criterion_mae_lambda = 10.0

def calculate_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold_dbz: float = 15.0,
) -> Dict[str, float]:
    """Compute SmoothL1, MAE, TOTAL, SSIM, CSI for a batch of sequences."""
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()
    sl1 = F.smooth_l1_loss(preds, targets)
    mae = F.l1_loss(preds, targets)
    total = sl1 * criterion_mae_lambda
    p = preds.numpy().squeeze()
    t = targets.numpy().squeeze()
    p = p * 0.5 + 0.5
    t = t * 0.5 + 0.5
    p_dbz = np.clip(p * 70.0, 0, 70)
    t_dbz = np.clip(t * 70.0, 0, 70)
    ssim_values = []
    for b in range(p_dbz.shape[0]):
        for tt in range(p_dbz.shape[1]):
            dr = max(t_dbz[b, tt].max() - t_dbz[b, tt].min(), 1e-6)
            if np.std(t_dbz[b, tt]) < 1e-6 or np.std(p_dbz[b, tt]) < 1e-6:
                ssim_values.append(1.0)
            else:
                ssim_values.append(
                    ssim(t_dbz[b, tt], p_dbz[b, tt], data_range=dr, win_size=5, channel_axis=None)
                )
    ssim_val = float(np.mean(ssim_values)) if ssim_values else 0.0
    p_bin = (p_dbz > threshold_dbz).astype(np.uint8)
    t_bin = (t_dbz > threshold_dbz).astype(np.uint8)
    cm = confusion_matrix(t_bin.flatten(), p_bin.flatten(), labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
    csi = tp / (tp + fp + fn + 1e-10)
    return {
        "TOTAL": float(total),
        "SmoothL1": float(sl1),
        "MAE": float(mae),
        "SSIM": float(ssim_val),
        "CSI": float(csi),
    }

def evaluate(model, loader, device, pred_length: int, threshold_dbz: float = 15.0) -> Tuple[Dict[str, float], np.ndarray]:
    """Evaluate model on a DataLoader and return average metrics + confusion matrix."""
    model.eval()
    agg = {"MAE": 0.0, "SSIM": 0.0, "CSI": 0.0, "SmoothL1": 0.0, "TOTAL": 0.0}
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                inputs, targets, mask, _ = batch
            else:
                inputs, targets, mask = batch
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs, _ = model(inputs, pred_length)
            m = calculate_metrics(outputs, targets, threshold_dbz=threshold_dbz)
            for k in agg:
                agg[k] += float(m[k])
            preds = outputs.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()
            preds = preds * 0.5 + 0.5
            targets_np = targets_np * 0.5 + 0.5
            preds_dbz = np.clip(preds * 70.0, 0, 70)
            targets_dbz = np.clip(targets_np * 70.0, 0, 70)
            preds_bin = (preds_dbz > threshold_dbz).astype(np.uint8)
            targets_bin = (targets_dbz > threshold_dbz).astype(np.uint8)
            all_preds.extend(preds_bin.flatten())
            all_targets.extend(targets_bin.flatten())
    for k in agg:
        agg[k] /= float(max(1, len(loader)))
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
    return agg, cm
