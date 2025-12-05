import os
import argparse
import re
import datetime as dt
from typing import List

import torch

from rainpred.model import RainPredRNN
from rainpred.geo_io import load_sequence_from_dir, save_predictions_as_geotiff

# ----------------------------------------------------------------------
# File-name handling for time-consistent output names
# ----------------------------------------------------------------------
# We expect names like:
#   rdr0_d01_YYYYMMDDZhhmm_VMI.tiff
# i.e.:
#   <prefix>YYYYMMDDZhhmm<suffix>
FILENAME_RE = re.compile(
    r"^(?P<prefix>.+_)"      # everything up to and including the last "_"
    r"(?P<date>\d{8})"       # YYYYMMDD
    r"Z"
    r"(?P<hour>\d{2})"       # hh
    r"(?P<minute>\d{2})"     # mm
    r"(?P<suffix>.*)$"       # rest of the name (e.g. "_VMI.tiff")
)


# ----------------------------------------------------------------------
# Device selection
# ----------------------------------------------------------------------
def get_device(force_cpu: bool = False) -> torch.device:
    """
    Decide which device (CPU or GPU) should be used.
    """
    if (not force_cpu) and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------------------------------------------------------
# Checkpoint loading (compatible with train.py)
# ----------------------------------------------------------------------
def load_model(
    checkpoint_path: str,
    device: torch.device,
    pred_length: int,
) -> torch.nn.Module:
    """
    Load a trained RainPredRNN model from a checkpoint file.

    Compatible with the checkpoints produced by train.py:
      - best_model.pth
      - last_checkpoint.pth

    It handles:
      * full nn.Module checkpoints
      * dict checkpoints with keys:
          - "model_state_dict" (train.py)
          - "state_dict"
      * plain state_dict checkpoints
      * possible 'module.' prefixes (DataParallel)

    Important: for PyTorch >= 2.6, we set weights_only=False to allow
    unpickling (old behavior).
    """
    ckpt = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,  # restore pre-2.6 behaviour
    )

    # Case 1: the checkpoint itself is already a full model
    if isinstance(ckpt, torch.nn.Module):
        model = ckpt.to(device)
        model.eval()
        return model

    # Otherwise, we expect a dict or a raw state_dict
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            # Fallback: assume the dict is itself a state_dict
            state_dict = ckpt
    else:
        # Fallback: checkpoint is directly a state_dict
        state_dict = ckpt

    # Handle DataParallel prefixes if present
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module."):] : v for k, v in state_dict.items()}

    # Instantiate the model with the same hyperparameters as train.py
    # train.py uses:
    #   RainPredRNN(input_dim=1, num_hidden=256,
    #               max_hidden_channels=128,
    #               patch_height=16, patch_width=16,
    #               pred_length=pred_length)
    # All of those except pred_length are defaults, so we only pass pred_length.
    model = RainPredRNN(pred_length=pred_length)
    model.to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ----------------------------------------------------------------------
# Time-step / naming utilities
# ----------------------------------------------------------------------
def infer_time_step_minutes(file_list: List[str]) -> int:
    """
    Infer the time step (in minutes) between the last two frames in file_list
    using the FILENAME_RE pattern. If this fails, return a default of 10 min.
    """
    if len(file_list) < 2:
        return 10

    m2 = FILENAME_RE.match(os.path.basename(file_list[-1]))   # newest
    m1 = FILENAME_RE.match(os.path.basename(file_list[-2]))   # previous

    if not (m1 and m2):
        return 10

    t1 = dt.datetime.strptime(
        m1.group("date") + m1.group("hour") + m1.group("minute"),
        "%Y%m%d%H%M",
    )
    t2 = dt.datetime.strptime(
        m2.group("date") + m2.group("hour") + m2.group("minute"),
        "%Y%m%d%H%M",
    )
    delta = t2 - t1
    minutes = int(delta.total_seconds() / 60)

    return minutes if minutes > 0 else 10


def build_output_basenames(file_list: List[str], n_future: int) -> List[str]:
    """
    Build base output file names for n_future prediction steps.

    If input names match FILENAME_RE, extend the timestamp of the last frame
    forward by the inferred time step; otherwise fall back to "pred_XX.tif".
    """
    if not file_list:
        return [f"pred_{i:02d}.tif" for i in range(1, n_future + 1)]

    last_name = os.path.basename(file_list[-1])
    match = FILENAME_RE.match(last_name)

    if not match:
        return [f"pred_{i:02d}.tif" for i in range(1, n_future + 1)]

    prefix = match.group("prefix")
    suffix = match.group("suffix")
    date_str = match.group("date")
    hour_str = match.group("hour")
    minute_str = match.group("minute")

    base_time = dt.datetime.strptime(
        date_str + hour_str + minute_str,
        "%Y%m%d%H%M",
    )
    step_minutes = infer_time_step_minutes(file_list)

    out_names: List[str] = []
    for i in range(1, n_future + 1):
        future_time = base_time + dt.timedelta(minutes=step_minutes * i)
        future_date = future_time.strftime("%Y%m%d")
        future_hm = future_time.strftime("%H%M")
        fname = f"{prefix}{future_date}Z{future_hm}{suffix}"
        out_names.append(fname)

    return out_names


# ----------------------------------------------------------------------
# Inference helper
# ----------------------------------------------------------------------
def run_inference(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    n_future: int,
) -> torch.Tensor:
    """
    Run the model in inference mode for a single sequence.

    The data pipeline (rainpred.geo_io.load_sequence_from_dir) already returns
    a tensor of shape (B, T_in, C, H, W). We simply ensure it is on the right
    device and call:
        preds, _ = model(x, pred_length=n_future)

    Returns
    -------
    preds : torch.Tensor
        Tensor of shape (B, n_future, C, H, W).
    """
    x = input_tensor.to(device)

    # Accept a few common shapes and normalize to (B, T, C, H, W)
    if x.dim() == 3:         # (H, W, ?) – very unlikely here, but just in case
        x = x.unsqueeze(0).unsqueeze(0)   # -> (1, 1, H, W)
    if x.dim() == 4:         # (T, C, H, W)
        x = x.unsqueeze(0)               # -> (1, T, C, H, W)
    elif x.dim() == 5:
        pass  # already (B, T, C, H, W)
    else:
        raise ValueError(f"Unexpected input tensor shape: {tuple(x.shape)}")

    with torch.no_grad():
        preds, _ = model(x, pred_length=n_future)  # RainPredRNN forward

    # IMPORTANT: keep the batch dimension so that save_predictions_as_geotiff
    # can handle (B, T, C, H, W) correctly.
    return preds


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """
    Define and parse command-line arguments for the script.
    """
    parser = argparse.ArgumentParser(
        description=(
            "RainPredictor inference: predict future radar frames from a "
            "sequence of GeoTIFFs, preserving geo-referencing and naming "
            "outputs consistently with the input files."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint (e.g. checkpoints/best_model.pth).",
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory with input radar frames (.tif / .tiff).",
    )

    parser.add_argument(
        "--m",
        type=int,
        default=18,
        help="Number of input frames used for conditioning (m).",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=6,
        help="Number of future frames to predict (n).",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where predicted GeoTIFFs will be stored.",
    )

    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force inference on CPU even if a CUDA GPU is available.",
    )

    return parser.parse_args()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    """
    Main entry point: load model, read input sequence, run inference,
    and save predictions as GeoTIFFs.
    """
    args = parse_args()

    device = get_device(force_cpu=args.force_cpu)
    print(f"[predict] Using device: {device}")

    # Load model compatible with train.py
    model = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
        pred_length=args.n,
    )

    # Load sequence and metadata from directory
    # geo_io.load_sequence_from_dir returns:
    #   seq        : torch.Tensor, shape (B, T, C, H_pad, W_pad)
    #   files      : list of file paths (sorted)
    #   shape_info : (orig_height, orig_width, pad_h, pad_w)
    #   meta       : rasterio metadata dict (copied from first frame)
    seq, file_list, shape_info, meta = load_sequence_from_dir(
        args.input_dir,
        args.m,
    )
    print(f"[predict] Loaded {len(file_list)} input frames from {args.input_dir}")

    # Build output basenames from last input file
    out_basenames = build_output_basenames(file_list=file_list, n_future=args.n)

    # Run model
    preds = run_inference(
        model=model,
        input_tensor=seq,
        device=device,
        n_future=args.n,
    )

    # Save predictions as GeoTIFF, cropping, padding and restoring metadata
    saved_paths = save_predictions_as_geotiff(
        preds=preds,
        output_dir=args.output_dir,
        shape_info=shape_info,
        meta=meta,
        prefix="pred",
        as_dbz=True,
        out_names=out_basenames,
    )

    # Summary
    print(f"[predict] Checkpoint used: {args.checkpoint}")
    print(f"[predict] Predicted {len(saved_paths)} future frames (n={args.n}).")
    print(f"[predict] Saved GeoTIFF predictions under: {args.output_dir}")
    for path in saved_paths:
        print(f"  -> {os.path.basename(path)}")


if __name__ == "__main__":
    main()
