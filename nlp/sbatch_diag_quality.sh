#!/bin/bash

# Whitening-quality diagnostic on the LayerNorm checkpoints (no training).
# Submit with: sg network-research -c "sbatch sbatch_diag_quality.sh"

#SBATCH --account=network_research_swarch
#SBATCH --job-name=diag-quality
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.out
#SBATCH --time=0:45:00

set -euo pipefail

source "${HOME}/.bashrc"
source activate env_python39

cd "${SLURM_SUBMIT_DIR:-${HOME}/batch_whitening}/nlp/nano_gpt"

python diag_whitening_quality.py "${HOME}/.cache/out-leak-ln/ckpt.pt" \
    ../../data/shakespeare_char/val.bin 64 1024
python diag_whitening_quality.py "${HOME}/.cache/out-gpt2-1gpu-g2-ln/ckpt.pt" \
    "${HOME}/.cache/openwebtext/val.bin" 12 384
