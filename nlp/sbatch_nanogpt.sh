#!/bin/bash

# Example: sbatch sbatch_nanogpt.sh
# Single-GPU nanoGPT job for shared GPU nodes.

#SBATCH --account=network_research_swarch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.out
#SBATCH --time=1:00:00

set -euo pipefail

source "${HOME}/.bashrc"
source activate env_python39

cd "${SLURM_SUBMIT_DIR:-${HOME}/batch_whitening}"
export PYTHONPATH="${PYTHONPATH:-}:${HOME}/batch_whitening"

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Starting single-GPU nanoGPT training..."

srun --unbuffered python ./nlp/nano_gpt/train.py

echo "Job completed."
