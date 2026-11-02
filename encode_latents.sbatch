#!/bin/bash
#SBATCH --partition=AMD7V12-RTX4500-CCP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -J ldm_encode
#SBATCH --output=/home/CCP/semenchuk.aa/DiffusionLearn/runs/encode_%A_%a.log

# Латенты по частям, одна часть на один GPU:
#   sbatch --array=0-2 encode_latents.sbatch ~/DiffusionLearn_data/pretrain_v1
# Аргументы после пути к датасету уходят в encode_latents.py.

DATASET=$1; shift
cd /home/CCP/semenchuk.aa/DiffusionLearn
.venv/bin/python encode_latents.py --dataset "$DATASET" --shard "$SLURM_ARRAY_TASK_ID" --nshards "$SLURM_ARRAY_TASK_COUNT" "$@"
