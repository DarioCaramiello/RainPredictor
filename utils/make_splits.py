#!/usr/bin/env python3
import os
import glob
import math
import random
import argparse
import shutil

# Progress bar (tqdm); fallback to plain iterator if not available
try:
    from tqdm import tqdm
except ImportError:  # simple fallback
    def tqdm(iterable, **kwargs):
        return iterable


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split a dataset of .tiff files into train/val/test using symlinks or copies."
    )

    # Base paths
    parser.add_argument(
        "--root",
        default="data/rdr0/",
        help="Base directory (default: %(default)s)",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing input .tiff files. "
             "If unset, defaults to ROOT.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory where train/val/test splits will be created. "
             "If unset, defaults to ROOT_splits.",
    )

    # Pattern and ratios
    parser.add_argument(
        "--pattern",
        default="**/*.tiff",
        help="Glob pattern (relative to data-dir) to find files (default: %(default)s)",
    )
    parser.add_argument(
        "--ratios",
        nargs=3,
        type=float,
        metavar=("TRAIN", "VAL", "TEST"),
        default=(0.90, 0.09, 0.01),
        help="Train/val/test split ratios (must sum to 1.0). Default: 0.90 0.09 0.01",
    )

    # Behavior
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without modifying the filesystem (no cleanup, no symlinks/copies).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when shuffling files (default: %(default)s).",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling; keep files in sorted order.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of creating symlinks.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    root = args.root
    data_dir = args.data_dir or root
    out_dir  = args.out_dir  or f"{root.rstrip('/')}_splits"
    pattern  = args.pattern
    ratios   = tuple(args.ratios)
    dry_run  = args.dry_run

    # Collect files
    files = sorted(glob.glob(os.path.join(data_dir, pattern), recursive=True))
    if not files:
        raise SystemExit(f"No files found in {data_dir} using pattern '{pattern}'.")

    # Check ratios
    if not math.isclose(sum(ratios), 1.0, rel_tol=1e-6):
        raise SystemExit(f"Ratios must sum to 1.0, but sum to {sum(ratios):.6f}")

    # Shuffle or not
    if not args.no_shuffle:
        random.seed(args.seed)
        random.shuffle(files)

    # Compute split sizes
    n = len(files)
    n_train = int(math.floor(n * ratios[0]))
    n_val   = int(math.floor(n * ratios[1]))
    n_test  = n - n_train - n_val

    train_files = files[:n_train]
    val_files   = files[n_train:n_train + n_val]
    test_files  = files[n_train + n_val:]

    print(f"Total files: {n} | train: {len(train_files)}  val: {len(val_files)}  test: {len(test_files)}")
    print(f"Input directory:  {data_dir}")
    print(f"Output directory: {out_dir}")
    mode_str = "COPY" if args.copy else "SYMLINK"
    print(f"Mode: {mode_str}")
    if dry_run:
        print("DRY RUN: no directories will be removed and no symlinks/copies will be created.")

    # Prepare output dirs
    train_dir = os.path.join(out_dir, "train")
    val_dir   = os.path.join(out_dir, "val")
    test_dir  = os.path.join(out_dir, "test")

    # Cleanup existing split dirs (only if not dry-run)
    if not dry_run:
        for d in (train_dir, val_dir, test_dir):
            if os.path.exists(d):
                print(f"Removing existing directory: {d}")
                shutil.rmtree(d, ignore_errors=True)
    else:
        for d in (train_dir, val_dir, test_dir):
            if os.path.exists(d):
                print(f"DRY RUN: would remove existing directory: {d}")

    # Create fresh dirs
    for d in (train_dir, val_dir, test_dir):
        if not dry_run:  # in dry run we skip creating as well
            ensure_dir(d)

    # Symlink/copy creation preserving directory structure
    def link_or_copy_into(subset_files, subset_dir, desc):
        if not subset_files:
            return
        for src in tqdm(subset_files, desc=desc):
            rel = os.path.relpath(src, data_dir)
            dst_dir = os.path.join(subset_dir, os.path.dirname(rel))
            dst = os.path.join(dst_dir, os.path.basename(src))

            if dry_run:
                # Just show what would be done
                continue

            ensure_dir(dst_dir)

            if os.path.islink(dst) or os.path.exists(dst):
                continue

            try:
                if args.copy:
                    shutil.copy2(src, dst)
                else:
                    os.symlink(src, dst)
            except FileExistsError:
                # Race conditions or parallel runs; safe to ignore
                pass

    link_or_copy_into(train_files, train_dir, "train")
    link_or_copy_into(val_files,   val_dir,   "val")
    link_or_copy_into(test_files,  test_dir,  "test")

    print(f"Dataset split ready in: {out_dir}")
    print("Sample files (first 3):")
    print("  train:", train_files[:3])
    print("  val  :", val_files[:3])
    print("  test :", test_files[:3])


if __name__ == "__main__":
    main()

