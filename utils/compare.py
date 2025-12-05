#!/usr/bin/env python
"""
compare.py

Compare truth vs predicted radar TIFF/GeoTIFF images as side-by-side panels
for a sequence of timestamps, and compute quantitative verification metrics.

Features
--------
- Sequence mode between --start and --end timestamps (YYYYMMDDZhhmm)
- Truth vs Pred in 2 columns, one row per timestamp
- Optional custom radar palette JSON (--palette)
- Robust handling of NaN/inf and constant fields (no vmin > vmax crashes)
- Logging instead of print
- Colorbar placed outside the plot area to avoid overlapping images
- Frame-by-frame and overall metrics (MSE, MAE, Bias, Correlation)
- Metrics panel plotted at the bottom of the figure
"""

import argparse
import json
import logging
import os
import re
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# Optional GeoTIFF-aware reader (rasterio) with fallback to Pillow
try:
    import rasterio

    HAS_RASTERIO = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_RASTERIO = False

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_PIL = False

# Timestamp pattern like 20251202Z1810
TIMESTAMP_RE = re.compile(r"(\d{8}Z\d{4})")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(level_str: str) -> None:
    """Configure the root logger with a simple format."""
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Palette handling
# ---------------------------------------------------------------------------
def load_palette(path: Path):
    """
    Load a radar palette JSON file of the form:

    {
      "levels": [-10, -5, 0, 5, ...],
      "colors": ["#646464", "#04e9e7", ...],
      "label": "Reflectivity (dBZ)"
    }

    Returns
    -------
    cmap : matplotlib.colors.ListedColormap
    norm : matplotlib.colors.BoundaryNorm
    label: str
        Label for the colorbar (may be empty).
    """
    logger.info("Loading palette from %s", path)
    with path.open("r") as f:
        pal = json.load(f)

    levels = pal["levels"]
    colors = pal["colors"]
    label = pal.get("label", "")

    if len(colors) != len(levels):
        raise ValueError(
            f"Palette {path}: {len(colors)} colors but {len(levels)} levels."
        )

    # BoundaryNorm expects len(boundaries) = len(colors) + 1
    boundaries = np.array(levels + [levels[-1] + 1], dtype=float)

    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)

    logger.info(
        "Palette loaded: %d colors, min=%.3f, max=%.3f",
        len(colors),
        levels[0],
        levels[-1],
    )
    # Attach label to norm for later retrieval
    if label:
        setattr(norm, "label", label)

    return cmap, norm, label


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def extract_timestamp(path: Path):
    """
    Extract timestamp of the form YYYYMMDDZhhmm from file name.

    Parameters
    ----------
    path : Path

    Returns
    -------
    str or None
    """
    m = TIMESTAMP_RE.search(path.name)
    if m:
        return m.group(1)
    return None


def scan_directory_for_tiffs(dir_path: Path):
    """
    Recursively scan directory for .tif/.tiff files and build a mapping:

        timestamp -> Path

    where timestamp is of the form YYYYMMDDZhhmm extracted from the filename.
    """
    mapping: dict[str, Path] = {}
    logger.info("Scanning directory for TIFF files: %s", dir_path)
    for root, _, files in os.walk(dir_path):
        for fname in files:
            if not fname.lower().endswith((".tif", ".tiff")):
                continue
            fpath = Path(root) / fname
            ts = extract_timestamp(fpath)
            if ts is None:
                logger.debug("No timestamp found in file name: %s", fpath)
                continue
            mapping[ts] = fpath
    logger.info(
        "Found %d TIFF files with valid timestamps in %s", len(mapping), dir_path
    )
    return mapping


def read_tiff(path: Path) -> np.ndarray:
    """
    Read the first band of a TIFF/GeoTIFF as a float32 numpy array.
    Nodata values (if present) are converted to NaN.

    Parameters
    ----------
    path : Path

    Returns
    -------
    np.ndarray
        2D array with dtype float32
    """
    logger.debug("Reading TIFF: %s", path)

    if HAS_RASTERIO:
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                data[data == nodata] = np.nan
        return data

    if HAS_PIL:
        im = Image.open(path)
        arr = np.array(im, dtype=np.float32)
        if arr.ndim > 2:
            # Take first channel if multi-band
            arr = arr[..., 0]
        return arr

    raise RuntimeError(
        "Neither rasterio nor Pillow (PIL) is available. "
        "Install one of them to read TIFF files."
    )


# ---------------------------------------------------------------------------
# Pair building and global stats
# ---------------------------------------------------------------------------
def build_pairs(
    truth_dir: Path, pred_dir: Path, start: str | None, end: str | None
):
    """
    Build a list of (timestamp, truth_path, pred_path) sorted by timestamp.
    Restrict to timestamps between start and end (inclusive) if provided.
    """
    truth_map = scan_directory_for_tiffs(truth_dir)
    pred_map = scan_directory_for_tiffs(pred_dir)

    common_ts = sorted(set(truth_map.keys()) & set(pred_map.keys()))
    logger.info("Found %d common timestamps between truth and pred.", len(common_ts))

    if start:
        common_ts = [ts for ts in common_ts if ts >= start]
    if end:
        common_ts = [ts for ts in common_ts if ts <= end]

    if not common_ts:
        logger.error("No common frames found in the specified range.")
        return []

    logger.info(
        "Using %d frame pairs between %s and %s",
        len(common_ts),
        common_ts[0],
        common_ts[-1],
    )

    pairs = [(ts, truth_map[ts], pred_map[ts]) for ts in common_ts]
    return pairs


def compute_global_range(arrays: list[np.ndarray]):
    """
    Compute a NaN/inf-safe global min and max over a list of arrays.

    Returns
    -------
    vmin, vmax : float
    """
    global_vmin = float("inf")
    global_vmax = float("-inf")

    for arr in arrays:
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        local_min = float(finite.min())
        local_max = float(finite.max())
        global_vmin = min(global_vmin, local_min)
        global_vmax = max(global_vmax, local_max)

    if not np.isfinite(global_vmin) or not np.isfinite(global_vmax):
        logger.warning("No finite values found across all frames. Using [0, 1].")
        return 0.0, 1.0

    if global_vmin >= global_vmax:
        logger.warning(
            "Global vmin >= vmax (vmin=%.3f, vmax=%.3f). "
            "Expanding range artificially.",
            global_vmin,
            global_vmax,
        )
        global_vmax = global_vmin + 1e-6

    logger.info("Global data range: vmin=%.3f, vmax=%.3f", global_vmin, global_vmax)
    return global_vmin, global_vmax


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_frame_metrics(truth: np.ndarray, pred: np.ndarray):
    """
    Compute basic verification metrics between truth and prediction.

    Returns
    -------
    metrics : dict
        Keys: mse, mae, bias, corr
    """
    mask = np.isfinite(truth) & np.isfinite(pred)
    if not np.any(mask):
        return {"mse": np.nan, "mae": np.nan, "bias": np.nan, "corr": np.nan}

    t = truth[mask].ravel()
    p = pred[mask].ravel()
    diff = p - t

    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))

    if t.size > 1 and np.std(t) > 0.0 and np.std(p) > 0.0:
        corr = float(np.corrcoef(t, p)[0, 1])
    else:
        corr = np.nan

    return {"mse": mse, "mae": mae, "bias": bias, "corr": corr}


def compute_overall_metrics(truth_list: list[np.ndarray], pred_list: list[np.ndarray]):
    """
    Compute overall metrics across all frames.

    Returns
    -------
    metrics : dict
        Keys: mse, mae, bias, corr
    """
    all_t = []
    all_p = []
    for truth, pred in zip(truth_list, pred_list):
        mask = np.isfinite(truth) & np.isfinite(pred)
        if not np.any(mask):
            continue
        all_t.append(truth[mask].ravel())
        all_p.append(pred[mask].ravel())

    if not all_t:
        return {"mse": np.nan, "mae": np.nan, "bias": np.nan, "corr": np.nan}

    t = np.concatenate(all_t)
    p = np.concatenate(all_p)
    diff = p - t

    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))

    if t.size > 1 and np.std(t) > 0.0 and np.std(p) > 0.0:
        corr = float(np.corrcoef(t, p)[0, 1])
    else:
        corr = np.nan

    return {"mse": mse, "mae": mae, "bias": bias, "corr": corr}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_sequence(
    pairs: list[tuple[str, Path, Path]],
    save_path: Path | None,
    title: str | None,
    cmap,
    norm,
):
    """
    Plot a sequence of frame pairs (truth vs pred) as a 2-column figure.
    Each row corresponds to a timestamp.

    If norm is provided (e.g. BoundaryNorm from a palette), it is used.
    Otherwise global vmin/vmax are computed and used.

    Additionally, a bottom panel shows frame-by-frame metrics (RMSE, MAE, Bias).
    """
    logger.info("Sequence mode: plotting %d frame pairs.", len(pairs))

    # First pass: read data and cache them
    cached: list[tuple[str, np.ndarray, np.ndarray]] = []
    all_truth: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    for ts, truth_path, pred_path in pairs:
        truth_data = read_tiff(truth_path)
        pred_data = read_tiff(pred_path)
        cached.append((ts, truth_data, pred_data))
        all_truth.append(truth_data)
        all_pred.append(pred_data)

    # If no palette norm is provided, compute a global vmin/vmax
    if norm is None:
        global_vmin, global_vmax = compute_global_range(all_truth + all_pred)
    else:
        global_vmin = global_vmax = None  # not used

    nframes = len(cached)

    # Precompute metrics
    per_frame_metrics = []
    for ts, truth_data, pred_data in cached:
        m = compute_frame_metrics(truth_data, pred_data)
        per_frame_metrics.append(m)
        logger.debug(
            "Metrics %s: MSE=%.4f, MAE=%.4f, Bias=%.4f, Corr=%.4f",
            ts,
            m["mse"],
            m["mae"],
            m["bias"],
            m["corr"],
        )

    overall = compute_overall_metrics(all_truth, all_pred)
    logger.info(
        "Overall metrics: MSE=%.4f, MAE=%.4f, Bias=%.4f, Corr=%.4f",
        overall["mse"],
        overall["mae"],
        overall["bias"],
        overall["corr"],
    )

    # ------------------------------------------------------------------
    # Figure layout using GridSpec: nframes rows for images + 1 row for metrics
    # ------------------------------------------------------------------
    import matplotlib.gridspec as gridspec

    height_ratios = [1.0] * nframes + [0.6]
    fig_height = max(4, 2 * nframes + 2)
    fig_width = 9

    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True)
    gs = gridspec.GridSpec(
        nrows=nframes + 1,
        ncols=2,
        figure=fig,
        height_ratios=height_ratios,
    )

    axes_img = np.empty((nframes, 2), dtype=object)

    last_im = None

    # Image panels
    for row_idx, (ts, truth_data, pred_data) in enumerate(cached):
        ax_truth = fig.add_subplot(gs[row_idx, 0])
        ax_pred = fig.add_subplot(gs[row_idx, 1])
        axes_img[row_idx, 0] = ax_truth
        axes_img[row_idx, 1] = ax_pred

        if norm is not None:
            im_truth = ax_truth.imshow(truth_data, cmap=cmap, norm=norm)
            im_pred = ax_pred.imshow(pred_data, cmap=cmap, norm=norm)
        else:
            im_truth = ax_truth.imshow(
                truth_data, cmap=cmap, vmin=global_vmin, vmax=global_vmax
            )
            im_pred = ax_pred.imshow(
                pred_data, cmap=cmap, vmin=global_vmin, vmax=global_vmax
            )

        ax_truth.set_title(f"Truth {ts}")
        ax_pred.set_title(f"Pred {ts}")

        ax_truth.axis("off")
        ax_pred.axis("off")

        # Annotate per-frame metrics on the left plot as text
        m = per_frame_metrics[row_idx]
        metric_text = (
            f"RMSE={np.sqrt(m['mse']):.2f}\n"
            f"MAE={m['mae']:.2f}\n"
            f"Bias={m['bias']:.2f}\n"
            f"R={m['corr']:.2f}"
        )
        ax_truth.text(
            0.01,
            0.02,
            metric_text,
            transform=ax_truth.transAxes,
            fontsize=8,
            va="bottom",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

        last_im = im_pred or im_truth

    # Metrics panel (bottom row, spanning both columns)
    ax_metrics = fig.add_subplot(gs[-1, :])

    frame_labels = [ts[-4:] for ts, _, _ in cached]  # use hhmm as short label
    rmse_vals = [np.sqrt(m["mse"]) if np.isfinite(m["mse"]) else np.nan for m in per_frame_metrics]
    mae_vals = [m["mae"] for m in per_frame_metrics]
    bias_vals = [m["bias"] for m in per_frame_metrics]

    x = np.arange(nframes)

    ax_metrics.plot(x, rmse_vals, marker="o", label="RMSE")
    ax_metrics.plot(x, mae_vals, marker="s", label="MAE")
    ax_metrics.plot(x, bias_vals, marker="^", label="Bias")

    ax_metrics.set_xticks(x)
    ax_metrics.set_xticklabels(frame_labels, rotation=45)
    ax_metrics.set_xlabel("Time (hhmm)")
    ax_metrics.set_ylabel("Error (dBZ)")
    ax_metrics.grid(True, linestyle="--", alpha=0.3)
    ax_metrics.legend(loc="upper right", fontsize=8)

    # Add overall metrics as a text box
    overall_text = (
        f"Overall: RMSE={np.sqrt(overall['mse']):.2f}, "
        f"MAE={overall['mae']:.2f}, Bias={overall['bias']:.2f}, "
        f"R={overall['corr']:.2f}"
    )
    ax_metrics.text(
        0.01,
        0.95,
        overall_text,
        transform=ax_metrics.transAxes,
        fontsize=9,
        va="top",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    # Colorbar on the right, outside the image panels
    if last_im is not None:
        cbar = fig.colorbar(
            last_im,
            ax=list(axes_img.ravel()),
            orientation="vertical",
            fraction=0.04,
            pad=0.02,
        )
        label = getattr(norm, "label", None) if norm is not None else None
        if label:
            cbar.set_label(label)

    # Global title (placed slightly high to avoid overlap)
    if title:
        fig.suptitle(title, fontsize=14, y=0.995)

    # Save or show
    if save_path is not None:
        logger.info("Saving figure to %s", save_path)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        logger.info("Displaying figure interactively.")
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare truth vs predicted radar TIFF/GeoTIFF sequences."
    )
    parser.add_argument(
        "--truth-dir",
        "--real-dir",  # backward-compatible alias
        dest="truth_dir",
        required=True,
        help="Directory containing truth (ground-truth) TIFF/GeoTIFF frames.",
    )
    parser.add_argument(
        "--pred-dir",
        required=True,
        help="Directory containing predicted TIFF/GeoTIFF frames.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start timestamp (YYYYMMDDZhhmm), inclusive.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End timestamp (YYYYMMDDZhhmm), inclusive.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Figure title.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help=(
            "Path to save the output figure (e.g., out.png). "
            "If omitted, the figure is shown interactively."
        ),
    )
    parser.add_argument(
        "--palette",
        type=str,
        default=None,
        help=(
            "Path to palette JSON file (levels + colors). "
            "If omitted, the default 'viridis' colormap is used."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR). Default: INFO.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level)

    truth_dir = Path(args.truth_dir).resolve()
    pred_dir = Path(args.pred_dir).resolve()

    if not truth_dir.is_dir():
        logger.error("truth-dir does not exist or is not a directory: %s", truth_dir)
        raise SystemExit(1)
    if not pred_dir.is_dir():
        logger.error("pred-dir does not exist or is not a directory: %s", pred_dir)
        raise SystemExit(1)

    # Palette or default colormap
    if args.palette is not None:
        cmap, norm, _ = load_palette(Path(args.palette))
    else:
        cmap = plt.get_cmap("viridis")
        norm = None

    pairs = build_pairs(truth_dir, pred_dir, args.start, args.end)
    if not pairs:
        raise SystemExit(1)

    save_path = Path(args.save).resolve() if args.save else None
    plot_sequence(
        pairs=pairs,
        save_path=save_path,
        title=args.title,
        cmap=cmap,
        norm=norm,
    )


if __name__ == "__main__":
    main()
