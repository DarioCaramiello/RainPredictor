# RainPredictor – Hybrid Training + Inference (GeoTIFF, Full Resolution)

This project implements a radar nowcasting model (`RainPredRNN`) that:

- Trains on GeoTIFF radar images at native resolution (no resize).
- Uses zero-padding (bottom/right) to satisfy UNet + Transformer patch constraints.
- Preserves georeferencing (CRS + affine transform) when saving predictions.
- Provides:
  - `train.py` for training,
  - `predict.py` for m→n inference,
  - `compare.py` for quick visual comparison in lon/lat.

## 1. Example Radar-style Naming

Example input frames:

```text
rdr0_d02_20251202Z1510_VMI.tiff
rdr0_d02_20251202Z1520_VMI.tiff
...
rdr0_d02_20251202Z1800_VMI.tiff
```

If you run:

```bash
python predict.py \
  --checkpoint checkpoints/best_model.pth \
  --input-dir /data/radar_seq \
  --output-dir /data/preds \
  --m 18 \
  --n 6
```

The code:

- Infers the time step from the last two inputs (e.g., 10 minutes).
- Builds the next `n` timestamps.
- Produces future filenames:

```text
rdr0_d02_20251202Z1810_VMI.tiff
rdr0_d02_20251202Z1820_VMI.tiff
...
```

All outputs are full-resolution GeoTIFFs with CRS and transform identical to the inputs.

## 2. Training Example

Dataset layout:

```text
/data/rdr0_splits/
├─ train/
│  ├─ rdr0_d02_*.tiff
├─ val/
│  ├─ rdr0_d02_*.tiff
```

Train with:

```bash
python train.py \
  --data-path /data/rdr0_splits \
  --epochs 20 \
  --batch-size 4 \
  --lr 1e-3 \
  --num-workers 8 \
  --pred-length 6
```

Use `--small-debug` to run a tiny subset for quick testing.

TensorBoard:

```bash
tensorboard --logdir runs
```

## 3. Inference Example

After training, run inference on a sequence directory:

```bash
python predict.py \
  --checkpoint checkpoints/best_model.pth \
  --input-dir /data/radar_seq \
  --output-dir /data/preds \
  --m 18 \
  --n 6
```

Options:

- `--pattern .tif` (default) or `.tiff` depending on files.
- `--cpu` to force CPU inference.

## 4. Visual Comparison in Physical Coordinates

```bash
python compare.py \
  --input /data/radar_seq/rdr0_d02_20251202Z1800_VMI.tiff \
  --pred  /data/preds/rdr0_d02_20251202Z1810_VMI.tiff \
  --title "t=1800 vs t+1 prediction"
```

This script:

- Uses rasterio to get transform and extent.
- Plots input and predicted frames side-by-side in lon/lat.
- Adds colorbars for quick qualitative assessment.

## 5. Reproducibility Notes

- `set_seed(15)` is called in the data and training code.
- Hyperparameters can be controlled from:
  - `rainpred/config.py` (defaults),
  - CLI flags in `train.py` and `infer_sequence.py`.
- All GeoTIFF outputs keep the original CRS and transform so that they can be
  consumed by GIS software or downstream hydrological models directly.
