#!/bin/bash
#SBATCH --job-name=rainpredtrain
#SBATCH --partition=h100gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

THIS_DIR=$PWD

set -euo pipefail

# --- Venv ---
module load gcc-9.9.1
module load openssl/3.6.0
module load python/3.11.14
module load cuda/12.4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID

source ./venv/bin/activate

echo "=== NODE/GPU INFO ==="
hostname
echo "Partition: $SLURM_JOB_PARTITION"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || true
echo "====================="


python -u train.py --data-path $THIS_DIR/data/rdr0_splits --val-preview-root $THIS_DIR/previews --runs-dir $THIS_DIR/runs --checkpoint-dir $THIS_DIR/checkpoints --epochs 10 --batch-size 4 --lr 1e-3 --num-workers 8

echo "Training ended: $(date)"
