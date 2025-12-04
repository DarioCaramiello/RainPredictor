# Import time to benchmark throughput
import time
# Import numpy for seeding and simple math
import numpy as np
# Import torch for DL operations
import torch

def set_seed(seed: int = 15) -> None:
    """Set random seeds for numpy and torch for better reproducibility."""
    # Seed numpy RNG
    np.random.seed(seed)
    # Seed torch CPU RNG
    torch.manual_seed(seed)
    # If CUDA is available, seed GPU RNGs
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Allow cuDNN to auto-tune kernels (fast, but not bitwise deterministic)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def hms(sec: float) -> str:
    """Convert seconds into HH:MM:SS string."""
    # Cast to integer seconds
    sec = int(sec)
    # Compute hours
    h = sec // 3600
    # Compute minutes
    m = (sec % 3600) // 60
    # Remaining seconds
    s = sec % 60
    # Format with zero padding
    return f"{h:02d}:{m:02d}:{s:02d}"

def benchmark_train(loader, model, optimizer, device, criterion_sl1,
                    pred_length: int, scaler=None, warmup: int = 2, measure: int = 10) -> float:
    """Measure train throughput (batches/s) including forward+backward+step."""
    # Put model into train mode
    model.train()
    # Get iterator
    it = iter(loader)
    # Warmup iterations (not timed)
    for _ in range(warmup):
        try:
            inputs, targets, _ = next(it)
        except StopIteration:
            return 0.0
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs, _ = model(inputs, pred_length)
                loss = criterion_sl1(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs, _ = model(inputs, pred_length)
            loss = criterion_sl1(outputs, targets)
            loss.backward()
            optimizer.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    counted = 0
    for _ in range(measure):
        try:
            inputs, targets, _ = next(it)
        except StopIteration:
            break
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs, _ = model(inputs, pred_length)
                loss = criterion_sl1(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs, _ = model(inputs, pred_length)
            loss = criterion_sl1(outputs, targets)
            loss.backward()
            optimizer.step()
        counted += 1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return counted / dt if dt > 0 else 0.0

def benchmark_val(loader, model, device, pred_length: int,
                  warmup: int = 2, measure: int = 20) -> float:
    """Measure validation throughput (batches/s) with forward-only passes."""
    model.eval()
    it = iter(loader)
    with torch.no_grad():
        for _ in range(warmup):
            try:
                inputs, _, _ = next(it)
            except StopIteration:
                return 0.0
            inputs = inputs.to(device, non_blocking=True)
            _ = model(inputs, pred_length)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        counted = 0
        for _ in range(measure):
            try:
                inputs, _, _ = next(it)
            except StopIteration:
                break
            inputs = inputs.to(device, non_blocking=True)
            _ = model(inputs, pred_length)
            counted += 1
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return counted / dt if dt > 0 else 0.0
