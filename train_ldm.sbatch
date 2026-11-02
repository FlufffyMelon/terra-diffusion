#!/bin/bash
#SBATCH --partition=AMD7V12-RTX4500-CCP
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --gpus-per-node=3
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH -t 12:00:00
#SBATCH -J ldm_train
#SBATCH --output=/home/CCP/semenchuk.aa/DiffusionLearn/runs/slurm_%j.log

# Обучение латентной диффузии на Lightning. Один процесс на GPU, запуск через srun, узел один.
#   RUN=ldm_v1 DATA=~/DiffusionLearn_data/pretrain_v1 sbatch -w node211-17 train_ldm.sbatch
# Если упрется в лимит времени, запусти еще раз с тем же RUN: продолжит с last.ckpt.

PROJECTDIR=/home/CCP/semenchuk.aa/DiffusionLearn
cd "$PROJECTDIR"
echo "=== LDM $(date) | job $SLURM_JOB_ID | node $(hostname) | run $RUN | GPU $SLURM_NTASKS_PER_NODE ==="
srun .venv/bin/python train_ldm.py --data "$DATA" --out "runs/$RUN" --devices "$SLURM_NTASKS_PER_NODE" "$@"
echo "=== done $(date) ==="
