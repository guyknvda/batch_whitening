#!/bin/bash

# Example: sbatch -J g2-ln sbatch_gpt2_1gpu.sh --wandb_run_name=gpt2-1gpu-ln --normalization_type=layer_norm
# Single-GPU GPT-2 124M job; extra arguments are forwarded to train.py as overrides.
# 10 micro-steps of 12 x 1024 tokens per iteration: the same per-GPU work as the
# 4-GPU baseline (40 micro-steps split over 4 ranks), with a 4x smaller global batch.
# Checkpoints go to a per-run directory on lustre so a run can be resumed.

#SBATCH --account=network_research_swarch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.out
#SBATCH --time=4:00:00

set -euo pipefail

source "${HOME}/.bashrc"
source activate env_python39

cd "${SLURM_SUBMIT_DIR:-${HOME}/batch_whitening}"
export PYTHONPATH="${PYTHONPATH:-}:${HOME}/batch_whitening"

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Args: $*"

srun --unbuffered python ./nlp/nano_gpt/train.py --config_name=gpt2 \
    --gradient_accumulation_steps=10 \
    --out_dir="${HOME}/.cache/out-gpt2-1gpu-${SLURM_JOB_NAME:-local}" "$@"

echo "Job completed."
