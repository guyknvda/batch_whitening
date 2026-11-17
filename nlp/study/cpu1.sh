#!/bin/bash
# CPU cost of a BW layer. Submit from the repo root: sbatch -J cpu1 nlp/study/cpu1.sh
#SBATCH --account=network_research_swarch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.out
#SBATCH --time=0:15:00
source "${HOME}/.bashrc"; source activate env_python39
cd "${SLURM_SUBMIT_DIR}/nlp/nano_gpt"; export PYTHONPATH="${PYTHONPATH:-}:${HOME}/batch_whitening"
python -u cpu_profile_bw.py > ../study/logs/cpu-profile-bw.out 2>&1
echo "Job completed."
