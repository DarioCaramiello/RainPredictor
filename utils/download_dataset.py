#!/usr/bin/env python3

# Import standard libraries
import argparse                           # For parsing command-line arguments
import os                                 # For filesystem operations
from datetime import datetime, timedelta  # For dates and time intervals
from concurrent.futures import ThreadPoolExecutor, as_completed  # For parallel downloads
import logging                            # For logging to console and optional file
import hashlib                            # For checksum (SHA256) verification

# Third-party libraries (install via: pip install requests tqdm)
import requests                           # For HTTP file downloads
from tqdm import tqdm                     # For progress bar visualization


# Time step between products (in minutes)
TIME_STEP_MINUTES = 10


def parse_datetime_yyyymmddhhmm(s: str) -> datetime:
    """
    Convert a YYYYMMDDZhhmm string into a datetime object.
    Example: '20251202Z1510' → datetime(2025, 12, 2, 15, 10)
    """
    try:
        s = s.replace("Z", " ")
        return datetime.strptime(s, "%Y%m%d %H%M")
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid datetime '{s}'. Expected format: YYYYMMDDZhhmm (e.g. 20251202Z1510)"
        ) from e


def build_filename(dt: datetime) -> str:
    """
    Build the radar filename for a given datetime.
    Example: dt -> 'rdr0_d01_20251202Z1510_VMI.tiff'
    """
    timestamp = dt.strftime("%Y%m%dZ%H%M")
    return f"rdr0_d01_{timestamp}_VMI.tiff"


def build_url(dt: datetime, base_url: str) -> str:
    """
    Build the remote URL for a radar image at datetime `dt`.

    Format:
    base_url/YYYY/MM/DD/rdr0_d01_YYYYMMDDZhhmm_VMI.tiff
    """
    year = dt.strftime("%Y")
    month = dt.strftime("%m")
    day = dt.strftime("%d")
    filename = build_filename(dt)
    return f"{base_url}/{year}/{month}/{day}/{filename}"


def build_output_path(output_dir: str, dt: datetime) -> str:
    """
    Construct a local file path mirroring the server structure:
    output_dir/YYYY/MM/DD/rdr0_d01_YYYYMMDDZhhmm_VMI.tiff
    """
    year = dt.strftime("%Y")
    month = dt.strftime("%m")
    day = dt.strftime("%d")
    filename = build_filename(dt)

    dir_path = os.path.join(output_dir, year, month, day)
    os.makedirs(dir_path, exist_ok=True)

    return os.path.join(dir_path, filename)


def download_file(url: str, dest_path: str, timeout: float = 30.0, logger=None) -> bool:
    """
    Download a remote file from `url` and save it to `dest_path`.
    Returns True if successful, False otherwise.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        response = requests.get(url, stream=True, timeout=timeout)
    except requests.RequestException as e:
        logger.error(f"Request failed for %s: %s", url, e)
        return False

    if response.status_code == 404:
        logger.warning("Not found (404): %s", url)
        return False

    if not response.ok:
        logger.error("HTTP %s for %s", response.status_code, url)
        return False

    try:
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    except OSError as e:
        logger.error("Failed to write %s: %s", dest_path, e)
        return False

    logger.info("Saved %s -> %s", url, dest_path)
    return True


def iterate_datetimes(start: datetime, end: datetime, step_minutes: int):
    """
    Generate datetimes between start and end inclusive, stepping by `step_minutes`.
    """
    current = start
    delta = timedelta(minutes=step_minutes)

    while current <= end:
        yield current
        current += delta


def load_checksums(path: str, logger=None):
    """
    Load a checksum manifest file.

    Expected format per line (whitespace-separated):
        filename sha256hex

    Lines starting with '#' or empty lines are ignored.

    Returns a dict: { filename: sha256hex }
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    checksums = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    logger.warning("Skipping malformed checksum line: %s", line)
                    continue
                filename, sha = parts[0], parts[1]
                checksums[filename] = sha.lower()
    except OSError as e:
        logger.error("Failed to read checksum file %s: %s", path, e)
        return {}

    logger.info("Loaded %d checksum entries from %s", len(checksums), path)
    return checksums


def compute_sha256(path: str) -> str:
    """
    Compute SHA256 hex digest of the file at `path`.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checksum(path: str, expected_sha256: str, logger=None) -> bool:
    """
    Verify that the SHA256 of file `path` matches `expected_sha256`.
    Returns True if it matches, False otherwise.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        actual = compute_sha256(path)
    except OSError as e:
        logger.error("Failed to read %s for checksum: %s", path, e)
        return False

    if actual.lower() == expected_sha256.lower():
        logger.info("Checksum OK for %s", path)
        return True
    else:
        logger.error("Checksum MISMATCH for %s: expected %s, got %s",
                     path, expected_sha256, actual)
        return False


def download_one(
    dt: datetime,
    output_dir: str,
    base_url: str,
    skip_existing: bool,
    retries: int,
    checksums: dict | None,
    verify_checksums: bool,
    logger=None,
) -> bool:
    """
    Download (and optionally verify) the file for a single datetime.

    Implements:
    - optional skip if file already exists
    - retry logic
    - optional checksum verification using a checksums dict
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    url = build_url(dt, base_url)
    dest_path = build_output_path(output_dir, dt)
    filename = os.path.basename(dest_path)

    # If skip_existing is enabled and file already exists, just skip
    if skip_existing and os.path.exists(dest_path):
        logger.info("Skipping existing file: %s", dest_path)
        # If checksum verification is requested, we can still verify
        if verify_checksums and checksums and filename in checksums:
            return verify_checksum(dest_path, checksums[filename], logger=logger)
        return True

    # Number of attempts (at least 1)
    attempts = max(1, retries)

    for attempt in range(1, attempts + 1):
        logger.info("Downloading %s (attempt %d/%d)", url, attempt, attempts)
        ok = download_file(url, dest_path, logger=logger)
        if not ok:
            # Download failed; if this was not the last attempt, try again
            if attempt < attempts:
                continue
            else:
                return False

        # If we reach here, download succeeded
        if verify_checksums and checksums and filename in checksums:
            # Verify checksum
            if verify_checksum(dest_path, checksums[filename], logger=logger):
                return True
            else:
                # Checksum failed; if we have more attempts, retry download
                if attempt < attempts:
                    logger.warning(
                        "Retrying download for %s due to checksum mismatch", filename
                    )
                    continue
                else:
                    return False
        else:
            # No checksum verification requested or available; success
            return True

    # Should not reach this point, but for safety:
    return False


def setup_logger(log_file: str | None) -> logging.Logger:
    """
    Configure and return a logger that logs to console,
    and optionally also to a file if `log_file` is provided.
    """
    logger = logging.getLogger("rdr0_downloader")
    logger.setLevel(logging.INFO)

    # Prevent adding handlers multiple times if called repeatedly
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Optional file handler
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def main():
    # -----------------------------
    # Command-line argument parser
    # -----------------------------
    parser = argparse.ArgumentParser(
        description="Download rdr0 radar TIFF files between two datetimes."
    )

    parser.add_argument(
        "start",
        type=parse_datetime_yyyymmddhhmm,
        help="Start datetime YYYYMMDDHHMM",
    )
    parser.add_argument(
        "end",
        type=parse_datetime_yyyymmddhhmm,
        help="End datetime YYYYMMDDHHMM",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="downloads",
        help="Output directory (default: downloads)",
    )
    parser.add_argument(
        "--base-url",
        default="https://data.meteo.uniparthenope.it/instruments/rdr0",
        help="Base URL for remote data (default: rdr0 server)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print URLs without downloading.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel downloads.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads with --parallel (default: 4)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip downloads if the destination file already exists.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Maximum number of attempts per file (default: 3).",
    )
    parser.add_argument(
        "--log-file",
        help="Path to a log file. If set, logs are also written there.",
    )
    parser.add_argument(
        "--checksum-file",
        help=(
            "Path to a checksum manifest file (filename + SHA256). "
            "If provided, enables checksum verification."
        ),
    )

    args = parser.parse_args()

    # Setup logger (console + optional file)
    logger = setup_logger(args.log_file)

    # Validate datetime range
    if args.end < args.start:
        parser.error("End datetime must be >= start datetime")

    logger.info("Start datetime: %s", args.start)
    logger.info("End datetime:   %s", args.end)
    logger.info("Output directory: %s", args.output_dir)
    logger.info("Base URL: %s", args.base_url)
    logger.info("Parallel: %s, workers: %d", args.parallel, args.workers)
    logger.info("Skip existing: %s", args.skip_existing)
    logger.info("Retries per file: %d", args.retries)
    if args.log_file:
        logger.info("Logging to file: %s", args.log_file)

    # Load checksums if a manifest is provided
    checksums = None
    verify_checksums = False
    if args.checksum_file:
        checksums = load_checksums(args.checksum_file, logger=logger)
        verify_checksums = True if checksums else False
        logger.info("Checksum verification enabled: %s", verify_checksums)

    # Build list of datetimes
    datetimes = list(iterate_datetimes(args.start, args.end, TIME_STEP_MINUTES))
    if not datetimes:
        logger.info("No time steps to process.")
        return

    # Dry-run mode: just print URLs with progress bar
    if args.dry_run:
        for dt in tqdm(datetimes, desc="Dry-run (URLs)", unit="file"):
            url = build_url(dt, args.base_url)
            print(url)
        return

    # Sequential mode
    if not args.parallel:
        for dt in tqdm(datetimes, desc="Downloading", unit="file"):
            download_one(
                dt,
                args.output_dir,
                args.base_url,
                skip_existing=args.skip_existing,
                retries=args.retries,
                checksums=checksums,
                verify_checksums=verify_checksums,
                logger=logger,
            )
        return

    # Parallel mode
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                download_one,
                dt,
                args.output_dir,
                args.base_url,
                args.skip_existing,
                args.retries,
                checksums,
                verify_checksums,
                logger,
            ): dt
            for dt in datetimes
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Downloading (parallel)",
            unit="file",
        ):
            dt = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error("Exception for %s: %s", dt, e)


if __name__ == "__main__":
    main()

