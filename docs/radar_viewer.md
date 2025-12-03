# Radar / GeoTIFF Visualizer

This project provides a command-line tool to visualize **radar TIFF and GeoTIFF** images using false-color weather radar look-up tables (LUTs).  
The visualization configuration—including colors and dBZ intensity intervals—is defined in an external **palette JSON file**.

---

## Features

- Load **standard TIFF** radar images (via Pillow)
- Load **GeoTIFF** images (via `rasterio`, if installed)
- Apply custom false-color LUTs for reflectivity visualization
- Define **colors**, **dBZ levels**, and **colorbar label** in a JSON file
- Optional **band selection** for multi-band GeoTIFFs
- Automatic printing of **GeoTIFF metadata**:
  - CRS (coordinate reference system)
  - Resolution
  - Bounds (extent)
- Support for using GeoTIFF spatial extent in plots
- Optional display of axes for geographic / projected coordinates
- Optional saving to file instead of interactive display
- Fully configurable via command-line parameters

---

## Requirements

Python 3.8+ and the following Python packages:

```bash
pip install numpy pillow matplotlib
```

To enable **GeoTIFF** support, also install:

```bash
pip install rasterio
```

If `rasterio` is not installed, the script will fall back to Pillow and treat the input as a plain TIFF without georeferencing.

---

## Palette JSON Format

Create a JSON file (e.g., `palette.json`) with:

```json
{
  "levels": [-10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
  "colors": [
    "#646464", "#04e9e7", "#019ff4", "#0300f4",
    "#02fd02", "#01c501", "#008e00", "#fdf802",
    "#e5bc00", "#fd9500", "#fd0000", "#d40000",
    "#bc0000", "#f800fd", "#9854c6"
  ],
  "label": "Reflectivity (dBZ)"
}
```

- **levels**: boundaries (in dBZ) used for discrete color intervals  
- **colors**: color list (same length or one less than `levels`)  
- **label**: optional colorbar label (defaults to `"Reflectivity (dBZ)"`)

This palette design allows you to maintain **product-specific visualization styles** (e.g., CAPPI, MAX, QPE, etc.) per JSON file.

---

## Usage

### Basic usage (TIFF or GeoTIFF)

```bash
python radar_viewer.py -i input.tiff -p palette.json
```

If `input.tiff` is a GeoTIFF and `rasterio` is installed:

- The script reads **band 1** by default.
- It prints GeoTIFF metadata to stdout:
  - CRS
  - Resolution
  - Bounds
- The image is plotted using the correct **spatial extent**.

If `input.tiff` is a plain TIFF or `rasterio` is not available:

- The script falls back to Pillow.
- No georeferencing is used.
- The image is plotted in pixel coordinates with no axes (by default).

---

## Band Selection for GeoTIFF

For multi-band GeoTIFFs (e.g., different radar products per band), you can select the band:

```bash
python radar_viewer.py -i radar_geotiff.tif -p palette.json --band 2
```

- `--band` is **1-based** (1 = first band, 2 = second band, ...).
- If the specified band is invalid, `rasterio` will raise an error.

---

## GeoTIFF Metadata Printing

When a file is successfully opened as a GeoTIFF via `rasterio`, the script prints:

- **CRS** (e.g., `EPSG:4326`, projected CRS string, or WKT)
- **Resolution** (pixel size in x and y)
- **Bounds** (min/max x and y in CRS units)

Example output:

```text
GeoTIFF detected: radar_geotiff.tif
  CRS: EPSG:32633
  Resolution: (1000.0, 1000.0)
  Bounds: left=350000.0, right=650000.0, bottom=4510000.0, top=4810000.0
```

This helps verify that the raster is correctly georeferenced and that plotting with `extent` is consistent.

---

## Controlling Axes and Extent

By default:

- For GeoTIFFs, the spatial `extent` is used.
- Axes are **hidden** to mimic a “pure image” radar display.

To keep axes visible (recommended when you care about coordinates):

```bash
python radar_viewer.py -i radar_geotiff.tif -p palette.json --show-axes
```

This will display axis ticks, which correspond to the coordinate values in the GeoTIFF CRS.

---

## Other CLI Options

### Custom title

```bash
python radar_viewer.py -i input.tiff -p palette.json --title "Radar Frame"
```

### Disable colorbar

```bash
python radar_viewer.py -i input.tiff -p palette.json --no-colorbar
```

### Save output to file instead of showing

```bash
python radar_viewer.py -i input.tiff -p palette.json --save output.png
```

### High DPI output

```bash
python radar_viewer.py -i input.tiff -p palette.json --dpi 200
```

---

## Example

```bash
python radar_viewer.py     -i /path/to/frame_01_pred.tiff     -p palette.json     --title "Hi-WeFAI Radar Prediction"
```

Example with GeoTIFF, second band, axes enabled:

```bash
python radar_viewer.py     -i /path/to/radar_geotiff.tif     -p palette.json     --band 2     --show-axes     --title "Radar GeoTIFF Band 2"
```

---

## License

Apache 2.0 License.

