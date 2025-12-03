# Dataset Splitter for TIFF Files

This tool splits a directory of `.tiff` files into **train**, **validation**, and **test** subsets.
It preserves any subdirectory structure and supports two modes:

- **symlink mode** (default): fast, lightweight
- **copy mode** (`--copy`): physically copies the files

The script also provides:

- **train/val/test ratios configurable via CLI**
- **cleanup of old split directories**
- **optional shuffling with random seed**
- **tqdm progress bars**
- **dry-run mode** (no changes on disk)

## Installation

### Requirements

- Python 3.8+
- Recommended Python packages:
  ```
  pip install tqdm
  ```
If `tqdm` is not installed, the script still works but without progress bars.

## Basic Usage

### Default behavior

If your data are located in:

```
data/rdr0/
   └── *.tiff (possibly inside subfolders)
```

simply run:

```
python split_tiff.py
```

This creates:

```
data/rdr0_splits/
    ├── train/
    ├── val/
    └── test/
```

using **symlinks**, with default ratios:

```
train = 90%
val   = 9%
test  = 1%
```

## Command-Line Options

### Directory options

| Flag | Description |
|------|-------------|
| `--root PATH` | Base directory (default: `data/rdr0/`) |
| `--data-dir PATH` | Input directory with TIFF files (default: `--root`) |
| `--out-dir PATH` | Output directory for splits (default: `ROOT_splits`) |

### Behavior options

| Flag | What it does |
|------|---------------|
| `--copy` | Copy files instead of creating symlinks |
| `--dry-run` | Do not remove directories or create any files |
| `--no-shuffle` | Keep input file order instead of random shuffle |
| `--seed N` | Random seed used when shuffling (default: `42`) |

### Ratio and pattern options

| Flag | Description |
|------|-------------|
| `--ratios TRAIN VAL TEST` | Must sum to 1.0 (default: `0.90 0.09 0.01`) |
| `--pattern GLOB` | Glob pattern for input search (default: `**/*.tiff`) |

## Examples

### 1. Default: shuffle + symlinks + standard ratios

```
python split_tiff.py
```

### 2. Dry-run (no filesystem changes)

```
python split_tiff.py --dry-run
```

Shows what would be done, but does *not* modify anything.

### 3. Copy files instead of symlinks

```
python split_tiff.py --copy
```

### 4. Change train/val/test ratios

Split 80/10/10:

```
python split_tiff.py --ratios 0.8 0.1 0.1
```

### 5. Disable shuffling and keep sorted order

```
python split_tiff.py --no-shuffle
```

### 6. Custom input and output directories

```
python split_tiff.py \
    --data-dir /datasets/radar/raw \
    --out-dir /datasets/radar/splits \
    --copy
```

### 7. Using a different file pattern

```
python split_tiff.py --pattern "*.tif"
```

## Output Structure

If the input contains nested folders:

```
data/rdr0/
   site_A/2023/*.tiff
   site_B/2024/*.tiff
```

The script preserves structure:

```
data/rdr0_splits/
   train/
       site_A/2023/...
       site_B/2024/...
   val/
       site_A/2023/...
       site_B/2024/...
   test/
       site_A/2023/...
       site_B/2024/...
```

## Notes

- Symlink mode is recommended for large datasets (fast + minimal disk usage).
- Copy mode is recommended only if:
  - you need portability,
  - working on Windows without symlink permissions,
  - or training code cannot follow symlinks.

