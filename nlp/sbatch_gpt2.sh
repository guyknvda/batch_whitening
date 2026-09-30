#!/bin/bash

# Example: sbatch sbatch_gpt2.sh
# GPT-2 124M nanoGPT job using all 8 GPUs on one node.

#SBATCH --account=network_research_swarch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --output=%x_%j.out
#SBATCH --time=4:00:00

set -euo pipefail

source "${HOME}/.bashrc"
source activate env_python39

cd "${SLURM_SUBMIT_DIR:-${HOME}/batch_whitening}"
export PYTHONPATH="${PYTHONPATH:-}:${HOME}/batch_whitening"

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Starting 8-GPU GPT-2 nanoGPT training..."

srun --unbuffered torchrun --standalone --nproc_per_node=4 \
  ./nlp/nano_gpt/train.py --config_name=gpt2

echo "Job completed."
