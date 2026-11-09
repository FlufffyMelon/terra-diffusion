#!/bin/bash
#SBATCH --partition=AMD7V12-RTX4500-CCP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -J ldm_encode
#SBATCH --output=runs/encode_%A_%a.log

# Latents in shards, one shard per GPU:
#   sbatch --array=0-2 slurm/encode_latents.sbatch ~/DiffusionLearn_data/pretrain_v1
# Arguments after the dataset path go to encode_latents.py.

DATASET=$1; shift
cd "$SLURM_SUBMIT_DIR"   # submit from the project root
.venv/bin/python scripts/data/encode_latents.py --dataset "$DATASET" --shard "$SLURM_ARRAY_TASK_ID" --nshards "$SLURM_ARRAY_TASK_COUNT" "$@"
