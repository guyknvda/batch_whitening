#!/bin/bash
# Activation matrix product timing. Submit from the repo root: sbatch -J gemm1 nlp/study/gemm1.sh
#SBATCH --account=network_research_swarch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.out
#SBATCH --time=0:15:00
source "${HOME}/.bashrc"; source activate env_python39
cd "${SLURM_SUBMIT_DIR}/nlp/nano_gpt"; export PYTHONPATH="${PYTHONPATH:-}:${HOME}/batch_whitening"
python -u bench_gemm.py > ../study/logs/bench-gemm.out 2>&1; nvidia-smi --query-gpu=name,clocks.max.sm --format=csv >> ../study/logs/bench-gemm.out
echo "Job completed."
