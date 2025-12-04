# Radar Dataset Downloader

This tool downloads weather radar TIFF files from the **meteo@uniparthenope** data server.

It supports:

- Custom date ranges  
- Custom filename prefix and postfix  
- Parallel downloads  
- Retry logic  
- Skipping previously downloaded files  
- Optional SHA256 checksum verification  
- Optional logging to file  
- Dry-run mode  
- Full exit-status reporting  

---

## ## URL structure

Data are stored on:

```
https://data.meteo.uniparthenope.it/instruments/rdr0/YYYY/MM/DD/
```

Filenames follow the pattern:

```
{prefix}{YYYYMMDDZhhmm}{postfix}
```

Default:

```
prefix = rdr0_d02_
postfix = _VMI.tiff
```

Example filename:

```
rdr0_d02_20251202Z1510_VMI.tiff
```

---

# ## Usage

### Basic usage

Download images from *2025‑12‑02 15:10* to *2025‑12‑02 17:00*:

```bash
python download_dataset.py 202512021510 202512021700
```

Downloads files from:

```
https://data.meteo.uniparthenope.it/instruments/rdr0/YYYY/MM/DD/rdr0_d02_YYYYMMDDZhhmm_VMI.tiff
```

and saves them under:

```
downloads/YYYY/MM/DD/
```

---

# ## Custom prefix and postfix

Example:

```bash
python download_dataset.py 202512021510 202512021700     --prefix rdr0_d01_     --postfix _VMI.tiff
```

This produces filenames such as:

```
rdr0_d01_20251202Z1510_VMI.tiff
```

---

# ## Parallel download

To speed up large downloads:

```bash
python download_dataset.py 202512021510 202512021700     --parallel --workers 8
```

---

# ## Skip files already downloaded

If you want to resume a previous session:

```bash
python download_dataset.py 202512021510 202512021700     --skip-existing
```

---

# ## Retry failed downloads

Example: retry each file up to **5 times**:

```bash
python download_dataset.py 202512021510 202512021700     --retries 5
```

---

# ## Dry run

Print what would be downloaded **without downloading anything**:

```bash
python download_dataset.py 202512021510 202512021700 --dry-run
```

---

# ## Logging

Write logs to a file:

```bash
python download_dataset.py 202512021510 202512021700     --log-file download.log
```

---

# ## Checksum verification

If you have a SHA256 manifest file (format: `filename sha256hex`):

```bash
python download_dataset.py 202512021510 202512021700     --checksum-file checksums.txt
```

All downloaded files will be verified.

---

# ## Changing the base URL

To download from another server while keeping the same folder structure:

```bash
python download_dataset.py 202512021510 202512021700     --base-url https://data.meteo.uniparthenope.it/instruments/rdr0/
```

---

# ## Exit codes

- **0** — all files downloaded successfully  
- **1** — one or more files failed  
- **2** — internal error (rare)

---

# ## Example: Full featured run

```bash
python download_dataset.py 202512021510 202512021700     --base-url https://data.meteo.uniparthenope.it/instruments/rdr0/     --prefix rdr0_d02_     --postfix _VMI.tiff     --parallel --workers 8     --skip-existing     --retries 5     --log-file radar_download.log
```

---

# ## Author

Generated automatically by ChatGPT upon request.

