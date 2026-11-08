#!/bin/bash

# Example: sbatch -J rm-gs64 sbatch_nanogpt_exp.sh --bw_group_size=64 --wandb_run_name=rm-gs64
# Single-GPU nanoGPT job; all arguments are forwarded to train.py as overrides.
# Checkpoints go to node-local /tmp because the home filesystem is nearly full.

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
echo "Args: $*"

srun --unbuffered python ./nlp/nano_gpt/train.py \
    --out_dir="/tmp/out-shakespeare-char-${SLURM_JOB_ID:-local}" "$@"

echo "Job completed."
