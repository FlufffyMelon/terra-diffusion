#!/bin/bash
#SBATCH --partition=AMD7V12-RTX4500-CCP
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH -t 12:00:00
#SBATCH -J ldm_train
#SBATCH --output=runs/slurm_%j.log

# Latent diffusion training on Lightning. One process per GPU, launched via srun, a single node.
#   RUN=ldm_v1 DATA=~/DiffusionLearn_data/pretrain_v1 sbatch slurm/train_ldm.sbatch
# If it hits the time limit, launch it again with the same RUN: it continues from last.ckpt.

PROJECTDIR="$SLURM_SUBMIT_DIR"   # submit from the project root
cd "$PROJECTDIR"
echo "=== LDM $(date) | job $SLURM_JOB_ID | node $(hostname) | run $RUN | GPU $SLURM_NTASKS_PER_NODE ==="
srun .venv/bin/python scripts/train_ldm.py --data "$DATA" --out "runs/$RUN" --devices "$SLURM_NTASKS_PER_NODE" "$@"
echo "=== done $(date) ==="
