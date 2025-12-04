import os
import datetime
import argparse

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from rainpred.config import (
    DEVICE,
    NUM_WORKERS,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    PRED_LENGTH,
    USE_AMP,
    DATA_PATH as DEFAULT_DATA_PATH,
    VAL_PREVIEW_ROOT as DEFAULT_VAL_PREVIEW_ROOT,
    RUNS_DIR as DEFAULT_RUNS_DIR,
    CHECKPOINT_DIR as DEFAULT_CHECKPOINT_DIR,
)
from rainpred.utils import hms, benchmark_train, benchmark_val, set_seed
from rainpred.data import create_dataloaders
from rainpred.model import RainPredRNN
from rainpred.metrics import criterion_sl1, evaluate
from rainpred.train_utils import train_epoch, save_val_previews

def main():
    """Train RainPredRNN on GeoTIFF radar data with padding, no resize."""
    parser = argparse.ArgumentParser(
        description="Train RainPredRNN radar nowcasting model (GeoTIFF, padding only)."
    )
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH,
                        help=f"Dataset root containing train/ and val/ (default: {DEFAULT_DATA_PATH})")
    parser.add_argument("--val-preview-root", type=str, default=DEFAULT_VAL_PREVIEW_ROOT,
                        help="Folder for validation PNG previews.")
    parser.add_argument("--runs-dir", type=str, default=DEFAULT_RUNS_DIR,
                        help="TensorBoard runs root directory.")
    parser.add_argument("--checkpoint-dir", type=str, default=DEFAULT_CHECKPOINT_DIR,
                        help="Directory for model checkpoints.")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of epochs.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size.")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS, help="DataLoader workers.")
    parser.add_argument("--pred-length", type=int, default=PRED_LENGTH, help="Prediction length (frames).")
    parser.add_argument("--small-debug", action="store_true",
                        help="Use small subset of data for quick tests.")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to checkpoint (.pth) to resume training from.")
    parser.add_argument("--save-every", type=int, default=1,
                        help="Save the last checkpoint every N epochs (default: 1).")
    args = parser.parse_args()

    set_seed(15)
    data_path = os.path.abspath(args.data_path)
    val_preview_root = os.path.abspath(args.val_preview_root)
    runs_dir = os.path.abspath(args.runs_dir)
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    num_epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    lr = float(args.lr)
    num_workers = int(args.num_workers)
    pred_length = int(args.pred_length)
    small_debug = bool(args.small_debug)
    resume_from = args.resume_from
    save_every = int(args.save_every)
    start_epoch = 0

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(runs_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    train_writer = SummaryWriter(log_dir=os.path.join(log_dir, "Train"))
    val_writer = SummaryWriter(log_dir=os.path.join(log_dir, "Validation"))

    model = RainPredRNN(
        input_dim=1,
        num_hidden=256,
        max_hidden_channels=128,
        patch_height=16,
        patch_width=16,
        pred_length=pred_length,
    )

    # Move model to device and enable multi-GPU training if available.
    if DEVICE == "cuda":
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            print(f"Using torch.nn.DataParallel on {n_gpus} GPUs")
            model = torch.nn.DataParallel(model)
        model = model.to(DEVICE)
    else:
        model = model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)
    scaler = torch.amp.GradScaler("cuda",enabled=(USE_AMP and DEVICE == "cuda"))

    train_loader, val_loader = create_dataloaders(
        data_path=data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        pred_length=pred_length,
        small_debug=small_debug,
    )

    steps_train = len(train_loader)
    steps_val = len(val_loader)

    best_val = float("inf")

    # ----- Optional: resume training from a checkpoint -----
    if resume_from is not None:
        if os.path.isfile(resume_from):
            print(f"Resuming training from checkpoint: {resume_from}")
            ckpt = torch.load(resume_from, map_location=DEVICE)

            # Extract the model state dict (handle plain or dict checkpoints).
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            else:
                state_dict = ckpt

            # Handle checkpoints saved from DataParallel (keys starting with 'module.').
            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {k[len("module.") :]: v for k, v in state_dict.items()}

            # Load weights into the underlying model if wrapped in DataParallel.
            model_to_load = model.module if isinstance(model, torch.nn.DataParallel) else model
            model_to_load.load_state_dict(state_dict)

            # Restore optimizer state if present.
            if isinstance(ckpt, dict) and "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])

            # Restore best_val and starting epoch if present.
            if isinstance(ckpt, dict):
                if "best_val" in ckpt:
                    best_val = ckpt["best_val"]
                elif "metrics" in ckpt and "TOTAL" in ckpt["metrics"]:
                    best_val = ckpt["metrics"]["TOTAL"]

                if "epoch" in ckpt:
                    start_epoch = ckpt["epoch"] + 1

            print(f"Resumed from epoch {start_epoch}, best_val={best_val:.6f}")
        else:
            print(f"WARNING: --resume-from path not found: {resume_from}")

    bps_train = benchmark_train(
        train_loader, model, optimizer, DEVICE, criterion_sl1,
        pred_length=pred_length, scaler=scaler, warmup=2, measure=10,
    )
    bps_val = benchmark_val(
        val_loader, model, DEVICE, pred_length=pred_length, warmup=2, measure=50,
    )

    eta_train = steps_train / bps_train if bps_train > 0 else float("inf")
    eta_val = steps_val / bps_val if bps_val > 0 else float("inf")

    print(f"[Benchmark] Train: ~{bps_train:.2f} batch/s | steps/epoch={steps_train} | ETA epoch ≈ {hms(eta_train)}")
    print(f"[Benchmark] Val  : ~{bps_val:.2f} batch/s | steps/val  ={steps_val}   | ETA val   ≈ {hms(eta_val)}")

    os.makedirs(val_preview_root, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train_loss = train_epoch(
            model, train_loader, optimizer, DEVICE, scaler=scaler, pred_length=pred_length
        )
        val_metrics, conf_matrix = evaluate(
            model, val_loader, DEVICE, pred_length=pred_length, threshold_dbz=15.0
        )
        scheduler.step(val_metrics["TOTAL"])

        print(f"\tTrain Loss: {train_loss:.4f}")
        print("\tVal " + ", ".join([f"{k}: {float(v):.4f}" for k, v in val_metrics.items()]))

        train_writer.add_scalar("Loss", train_loss, epoch)
        for k, v in val_metrics.items():
            tag = "Loss" if k.lower() == "total" else k
            val_writer.add_scalar(tag, float(v), epoch)

        save_val_previews(
            model, val_loader, DEVICE, val_preview_root, epoch, pred_length=pred_length, overwrite=False
        )

        if val_metrics["TOTAL"] < best_val:
            best_val = val_metrics["TOTAL"]
            model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": val_metrics,
                    "confusion_matrix": conf_matrix,
                    "best_val": best_val,
                },
                os.path.join(checkpoint_dir, "best_model.pth"),
            )

        # Save the last checkpoint regularly for potential restarts.
        if ((epoch + 1) % save_every == 0) or (epoch + 1 == num_epochs):
            model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
            last_ckpt_path = os.path.join(checkpoint_dir, "last_checkpoint.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": val_metrics,
                    "confusion_matrix": conf_matrix,
                    "best_val": best_val,
                },
                last_ckpt_path,
            )

    train_writer.close()
    val_writer.close()
    print("Training completed.")

if __name__ == "__main__":
    main()
