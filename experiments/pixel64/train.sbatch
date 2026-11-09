#!/bin/bash
#SBATCH --partition=AMD7V12-V100-CCP
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 720
#SBATCH -J ddpm_cat
#SBATCH --output=runs/slurm_%j.log

# Training a diffusion model on cat photos (64x64, VP-SDE, U-Net).
# Launch:              sbatch experiments/pixel64/train.sbatch   (from the repository root)
# Another name and model:  RUN=cat64_small sbatch experiments/pixel64/train_ada.sbatch --model unet_small
# Any arguments after the script name go to train.py.
# If it hits the time limit, just launch it again with the same RUN: it continues from last.pt.

PROJECTDIR="$SLURM_SUBMIT_DIR"   # submit from the project root
VENV=${VENV:-${PROJECTDIR}/.venv}
RUN=${RUN:-cat64}
OUTDIR=${PROJECTDIR}/runs/${RUN}

mkdir -p "$OUTDIR"
source "${VENV}/bin/activate"
export OMP_NUM_THREADS=1   # parallelize data loading with DataLoader workers, not with threads inside them

echo "=== DDPM cat $(date) | job $SLURM_JOB_ID | node $(hostname) | run $RUN ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python "${PROJECTDIR}/experiments/pixel64/train.py" \
    --data-dir "${PROJECTDIR}/dataset_unique" \
    --out-dir "$OUTDIR" \
    --workers $(( ${SLURM_CPUS_PER_TASK:-16} - 2 )) \
    --resume \
    "$@"
echo "=== done $(date) ==="
