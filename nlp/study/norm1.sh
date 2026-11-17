#!/bin/bash
# Post-BW normalization kernels. Submit from the repo root: sbatch -J norm1 nlp/study/norm1.sh
#SBATCH --account=network_research_swarch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.out
#SBATCH --time=0:15:00
source "${HOME}/.bashrc"; source activate env_python39
cd "${SLURM_SUBMIT_DIR}/nlp/nano_gpt"; export PYTHONPATH="${PYTHONPATH:-}:${HOME}/batch_whitening"
python -u bench_norm.py > ../study/logs/bench-norm.out 2>&1
echo "Job completed."
