import os
from typing import List

import numpy as np
from PIL import Image
import torch

from .metrics import criterion_sl1, criterion_mae_lambda

def train_epoch(model, loader, optimizer, device, scaler=None, pred_length: int = 6) -> float:
    """Run one training epoch and return average loss."""
    model.train()
    total_loss = 0.0
    for inputs, targets, _ in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs, _ = model(inputs, pred_length)
                loss = criterion_sl1(outputs, targets) * criterion_mae_lambda
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs, _ = model(inputs, pred_length)
            loss = criterion_sl1(outputs, targets) * criterion_mae_lambda
            loss.backward()
            optimizer.step()
        total_loss += float(loss.detach().cpu())
    return total_loss / max(1, len(loader))

def save_val_previews(
    model,
    val_loader,
    device,
    out_root: str,
    epoch: int,
    pred_length: int,
    overwrite: bool = False,
) -> None:
    """Save a few prediction/target PNG pairs from validation for quick visual checks."""
    model.eval()
    ep_dir = os.path.join(out_root, f"epoch_{epoch:03d}")
    out_pred = os.path.join(ep_dir, "predictions")
    out_targ = os.path.join(ep_dir, "targets")
    os.makedirs(out_pred, exist_ok=True)
    os.makedirs(out_targ, exist_ok=True)
    saved = 0
    max_to_save = 8
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 4:
                inputs, targets, mask, paths = batch
            else:
                inputs, targets, mask = batch
                paths = ["sample"] * targets.shape[1]
            inputs = inputs.to(device, non_blocking=True)
            outputs, _ = model(inputs, pred_length)
            preds = outputs.detach().cpu().numpy()
            targs = targets.detach().cpu().numpy()
            preds = preds * 0.5 + 0.5
            targs = targs * 0.5 + 0.5
            if preds.ndim == 5:
                preds = preds[0, :, 0]
                targs = targs[0, :, 0]
            T = preds.shape[0]
            for t in range(T):
                stem = os.path.splitext(os.path.basename(paths[t]))[0] if paths else f"frame_{t:03d}"
                pred_path = os.path.join(out_pred, f"{stem}_pred.png")
                targ_path = os.path.join(out_targ, f"{stem}_target.png")
                if (not overwrite) and os.path.exists(pred_path):
                    continue
                p = (preds[t] * 255.0).clip(0, 255).astype(np.uint8)
                g = (targs[t] * 255.0).clip(0, 255).astype(np.uint8)
                Image.fromarray(p).save(pred_path)
                Image.fromarray(g).save(targ_path)
                saved += 1
                if saved >= max_to_save:
                    return
