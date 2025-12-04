# Radar dataset downloader

This script downloads a sequence of weather radar TIFF images from a base URL,
organised in the following directory layout on the server:

`{base_url}/YYYY/MM/DD/<prefix>YYYYMMDDZhhmm<postfix>`

For example:

`https://data.meteo.uniparthenope.it/instruments/rdr0/2025/12/01/rdr0_d02_20251201Z1510_VMI.tiff`

The script supports:

- Date–range based download with a fixed 10-minute step.
- Sequential or parallel downloads.
- Skipping files that are already present locally.
- Retry logic on failures.
- Optional checksum verification via a manifest file.
- **Optional in-place resampling of downloaded images, preserving aspect ratio.**

## Usage

```bash
./download_dataset-2.py START_DATETIME END_DATETIME [options]
```

Where dates are expressed as:

- `START_DATETIME` = `YYYYMMDDHHMM`
- `END_DATETIME`   = `YYYYMMDDHHMM`

Example:

```bash
./download_dataset-2.py 202512010000 202512012359 \
    --output-dir data/rdr0_raw \
    --base-url https://data.meteo.uniparthenope.it/instruments/rdr0 \
    --prefix rdr0_d02_ \
    --postfix _VMI.tiff
```

## Main options

- `-o, --output-dir DIR`  
  Destination root directory (default: `downloads`).

- `--base-url URL`  
  Base URL of the radar dataset (default:
  `https://data.meteo.uniparthenope.it/instruments/rdr0`).

- `--prefix PREFIX`  
  Filename prefix (default: `rdr0_d02_`).

- `--postfix POSTFIX`  
  Filename postfix/extension (default: `_VMI.tiff`).

- `--dry-run`  
  Do not download anything; print the URLs that would be fetched.

- `--parallel`  
  Enable parallel downloads using a thread pool.

- `--workers N`  
  Number of parallel worker threads when `--parallel` is active (default: 4).

- `--skip-existing`  
  Skip files already present in the output directory.

- `--retries N`  
  Number of download attempts per file (default: 3).

- `--log-file PATH`  
  Also, write log messages to the given file.

- `--checksum-file PATH`  
  Path to a text file with `filename sha256` pairs used to verify downloads.

### New: image resampling

- `--resample FACTOR`  
  Resample each downloaded image in-place by the given factor, **preserving
  aspect ratio**. Width and height are both multiplied by `FACTOR`.

  Examples:
  - `--resample 0.5`  → image size reduced to 50% in each dimension.  
  - `--resample 2.0`  → image size doubled in each dimension.  
  - `--resample 1.0`  → no resampling (default).

Resampling is performed after a successful download (and optional checksum
verification) using Pillow (PIL). Images are overwritten in-place, preserving
their original format.

> **Note:** Resampling modifies the file contents, so it is **not allowed
> together with checksum verification** (`--checksum-file`). If you set a
> resample factor different from `1.0` and also provide `--checksum-file`, the
> script will exit with an error.

## Requirements

- Python 3.8+
- Required Python packages:
  - `requests`
  - `tqdm`
  - `Pillow` (only if `--resample` is used)

Install them with:

```bash
pip install requests tqdm Pillow
```

## Exit codes

- `0` on success (all requested files downloaded/skipped cleanly).
- `1` if at least one download (or verification/resampling) fails.

---

_Last update: 2025-12-04T16:29:20 UTC_
