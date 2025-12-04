# Import os to handle filesystem paths
import os
# Import torch to detect CUDA and select device
import torch

# Decide which device to use: GPU if available, else CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Default DataLoader worker count
NUM_WORKERS = 8
# Default training batch size
BATCH_SIZE = 4
# Enable pinned memory for faster host→device copies
PIN_MEMORY = True

# Default learning rate
LEARNING_RATE = 1e-3
# Default number of epochs
NUM_EPOCHS = 1
# Default number of prediction frames
PRED_LENGTH = 6
# Enable Automatic Mixed Precision on GPU
USE_AMP = True

# Default patch size for temporal transformer
PATCH_HEIGHT = 16
PATCH_WIDTH = 16

# Default dataset root (overridden by CLI)
DATA_PATH = os.path.abspath("/home/v.bucciero/data/instruments/rdr0_splits/")
# Default validation preview directory
VAL_PREVIEW_ROOT = os.path.abspath("/home/v.bucciero/data/instruments/rdr0_previews_hybrid")
# Default TensorBoard runs directory
RUNS_DIR = "runs"
# Default checkpoint directory
CHECKPOINT_DIR = "checkpoints"
