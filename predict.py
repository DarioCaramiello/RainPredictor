import os
import argparse
import re
import datetime as dt

import torch

from rainpred.model import RainPredRNN
from rainpred.geo_io import load_sequence_from_dir, save_predictions_as_geotiff

def get_device(force_cpu: bool = False) -> torch.device:
    """Return CUDA device if available (and not forced to CPU), otherwise CPU."""
    if (not force_cpu) and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def generate_future_filenames(input_files, n: int):
    """Generate n future radar-style filenames continuing the temporal sequence.

    Pattern assumed:
      <prefix>YYYYMMDDZHHMM<middle>.<ext>
    Example:
      rdr0_d01_20251202Z1510_VMI.tiff
    """
    if len(input_files) < 2:
        raise ValueError("Need at least 2 input files to infer time step.")
    last_base = os.path.basename(input_files[-1])
    prev_base = os.path.basename(input_files[-2])
    pattern = r"^(.*_)(\d{8})Z(\d{4})(.*)\.(tif|tiff)$"
    m_last = re.match(pattern, last_base)
    m_prev = re.match(pattern, prev_base)
    if not (m_last and m_prev):
        raise ValueError(f"Filenames do not match expected pattern: '{prev_base}', '{last_base}'")
    date_last = m_last.group(2)
    time_last = m_last.group(3)
    date_prev = m_prev.group(2)
    time_prev = m_prev.group(3)
    dt_last_str = f"{date_last}{time_last}"
    dt_prev_str = f"{date_prev}{time_prev}"
    dt_last = dt.datetime.strptime(dt_last_str, "%Y%m%d%H%M")
    dt_prev = dt.datetime.strptime(dt_prev_str, "%Y%m%d%H%M")
    delta = dt_last - dt_prev
    if delta.total_seconds() <= 0:
        raise ValueError(f"Non-positive time step inferred from '{prev_base}', '{last_base}'")
    prefix = m_last.group(1)
    middle = m_last.group(4)
    ext = m_last.group(5)
    future_names = []
    for k in range(1, n + 1):
        new_dt = dt_last + k * delta
        new_date_str = new_dt.strftime("%Y%m%d")
        new_time_str = new_dt.strftime("%H%M")
        new_name = f"{prefix}{new_date_str}Z{new_time_str}{middle}.{ext}"
        future_names.append(new_name)
    return future_names

def load_model(checkpoint_path: str, device: torch.device, pred_length: int) -> RainPredRNN:
    """Instantiate RainPredRNN and load weights from checkpoint."""
    model = RainPredRNN(
        input_dim=1,
        num_hidden=256,
        max_hidden_channels=128,
        patch_height=16,
        patch_width=16,
        pred_length=pred_length,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model

def main():
    """Run inference on a sequence of radar GeoTIFF frames."""
    parser = argparse.ArgumentParser(
        description=(
            "RainPred hybrid inference: predict future radar frames from a sequence, "
            "preserving GeoTIFF resolution and naming outputs like input frames."
        )
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint (best_model.pth).")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory with input radar frames (.tif/.tiff).")
    parser.add_argument("--m", type=int, default=18,
                        help="Number of input frames used for conditioning (m > n).")
    parser.add_argument("--n", type=int, default=6,
                        help="Number of future frames to predict (n < m).")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory where predicted GeoTIFFs will be stored.")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU inference even if CUDA is available.")
    parser.add_argument("--pattern", type=str, default=".tif",
                        help="Filename extension filter (default: .tif).")
    args = parser.parse_args()

    if not (args.m > args.n):
        raise ValueError(f"Constraint m > n not satisfied: m={args.m}, n={args.n}.")

    checkpoint_path = os.path.abspath(args.checkpoint)
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)

    device = get_device(force_cpu=args.cpu)
    model = load_model(checkpoint_path, device, pred_length=args.n)

    seq, file_list, shape_info, meta = load_sequence_from_dir(
        input_dir, m=args.m, pattern=args.pattern,
    )
    seq = seq.to(device)
    out_basenames = generate_future_filenames(file_list, args.n)

    with torch.no_grad():
        outputs, _ = model(seq, pred_length=args.n)

    saved_paths = save_predictions_as_geotiff(
        outputs,
        output_dir,
        shape_info,
        meta,
        prefix="pred",
        as_dbz=True,
        out_names=out_basenames,
    )

    print(f"Loaded {len(file_list)} input frames from: {input_dir}")
    print(f"Checkpoint used: {checkpoint_path}")
    print(f"Predicted {len(saved_paths)} future frames (n={args.n}).")
    print(f"Saved GeoTIFF predictions under: {output_dir}")
    for p in saved_paths:
        print(f"  -> {os.path.basename(p)}")

if __name__ == "__main__":
    main()
