#!/bin/bash
#SBATCH --partition=AMD7V12-V100-CCP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 720
#SBATCH -J ddpm_cat
#SBATCH --output=/home/CCP/semenchuk.aa/DiffusionLearn/runs/slurm_%j.log

# Обучение диффузионной модели на фото кошки (64x64, VP-SDE, U-Net).
# Запуск:              sbatch train.sbatch
# Другое имя и модель:  RUN=cat64_small sbatch train_ada.sbatch --model unet_small
# Любые аргументы после имени скрипта уходят в train.py.
# Если упрется в лимит времени, просто запусти еще раз с тем же RUN: продолжит с last.pt.

PROJECTDIR=/home/CCP/semenchuk.aa/DiffusionLearn
VENV=${VENV:-${PROJECTDIR}/.venv}
RUN=${RUN:-cat64}
OUTDIR=${PROJECTDIR}/runs/${RUN}

mkdir -p "$OUTDIR"
source "${VENV}/bin/activate"
export OMP_NUM_THREADS=1   # параллелим загрузку данных воркерами DataLoader, а не потоками внутри них

echo "=== DDPM cat $(date) | job $SLURM_JOB_ID | node $(hostname) | run $RUN ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python "${PROJECTDIR}/train.py" \
    --data-dir "${PROJECTDIR}/dataset_unique" \
    --out-dir "$OUTDIR" \
    --workers $(( ${SLURM_CPUS_PER_TASK:-16} - 2 )) \
    --resume \
    "$@"
echo "=== done $(date) ==="
