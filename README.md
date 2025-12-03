
# RainPredictor

## Overview
RainPredictor is an advanced deep learning system for radar-based precipitation nowcasting, combining a U-Net encoder/decoder architecture with a temporal Transformer. The model processes sequences of radar reflectivity images (TIFF format) to predict future precipitation patterns.

### Key Features
- **Advanced Architecture**: 
  - U-Net encoder/decoder for robust spatial feature extraction
  - Temporal Transformer for sophisticated sequence modeling
  - Hybrid design optimized for meteorological predictions
- **Data Processing**:
  - Input/Output: Single-channel radar TIFF images
  - Automated data normalization and augmentation
  - Comprehensive evaluation metrics (MAE, MSE, SSIM, CSI)
- **Technical Features**:
  - Mixed precision training (AMP) on CUDA
  - Flexible gradient accumulation
  - Cross-platform support (Linux, macOS, Windows)
  - Real-time TensorBoard logging
  - Extensive performance optimization options

### Dataset Structure
The dataset should be organized in three splits (train/val/test), each containing chronologically ordered TIFF frames. The data loader recursively scans for `*.tiff` files in each directory.

## 2. Model Architecture

### Network Components
- **Encoder Network**
  - U-Net downsampling path with Conv-BN-ReLU blocks
  - MaxPool operations for spatial feature hierarchy
  - Efficient feature extraction from radar images

- **Temporal Processing**
  - Transformer architecture for sequence modeling
  - Patch-wise embeddings for temporal attention
  - Advanced temporal dependency learning

- **Decoder Network**
  - U-Net upsampling path with skip connections
  - Feature refinement through progressive upsampling
  - High-resolution output reconstruction

### Data Processing Pipeline
- **Normalization**
  - Radar values clipped to [0,70] dBZ range
  - Linear scaling to [0,1] for stable training
  - Automated handling of missing/invalid values

- **Augmentation** (Training Only)
  - Spatial flips and random affine transformations
  - Implemented via torchio for efficiency
  - Configurable augmentation parameters

### Evaluation System
- **Core Metrics**
  - MAE (Mean Absolute Error)
  - MSE (Mean Squared Error)
  - SSIM (Structural Similarity Index)
  - CSI (Critical Success Index) with configurable dBZ threshold

- **Monitoring**
  - Real-time TensorBoard logging
  - Per-epoch training loss tracking
  - Validation metrics visualization
  - Model prediction samples

## 3. Installation and Setup

### Prerequisites
> **Recommended**: Use Conda (conda-forge) for consistent GDAL/rasterio installation

### Dependencies
- **Python Version**: 3.10-3.12
- **Core Libraries**:
  - PyTorch & torchvision
  - NumPy & scikit-image
  - scikit-learn
  - pytorch-msssim & einops
  - rasterio (GDAL-dependent)
  - Pillow, tensorboard, torchio

### Installation Guide

### Create env
```
conda create -n rainpredrnn2 python=3.11 -y
conda activate rainpredrnn2
```
## Install PyTorch
## Linux/Windows CUDA (choose version from pytorch.org if needed)
CPU-only: replace with 
```
pytorch torchvision cpuonly -c pytorch
```

```
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### Rasterio + deps (via conda-forge: safest)
```
conda install -c conda-forge rasterio gdal -y
```

### The rest via pip (or conda if you prefer)
```
pip install numpy scikit-image scikit-learn pytorch-msssim einops pillow tensorboard torchio
```
### 3.2 macOS (Apple Silicon)

Install PyTorch with MPS support (CPU/MPS):
```
conda create -n rainpredrnn2 python=3.11 -y
conda activate rainpredrnn2
conda install pytorch torchvision -c pytorch -y       # this enables MPS on macOS
conda install -c conda-forge rasterio gdal -y
pip install numpy scikit-image scikit-learn pytorch-msssim einops pillow tensorboard torchio
```
You can use "mps" as device (see §6 “Configuration”).

### 3.3 Windows notes
Prefer Conda: 
```
conda-forge provides compatible GDAL/rasterio builds.
```
For CUDA, install the matching PyTorch build (see pytorch.org “Get Started”).

## 4) Quick Start

### macOS (local)

### Linux cluster / Ubuntu


## Usage
* Training [link](docs/train.md)
* Inference [link](docs/predict.md)
* Dataset splitting [link](docs/make_splits.md)
* Radar image visualizer [link](docs/radar_viewer.md)

## License
Apache 2.0 License.

