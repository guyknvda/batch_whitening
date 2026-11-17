#!/bin/bash
# Host-device syncs in a BW step. Submit from the repo root: sbatch -J sync1 nlp/study/sync1.sh
#SBATCH --account=network_research_swarch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.out
#SBATCH --time=0:15:00
source "${HOME}/.bashrc"; source activate env_python39
cd "${SLURM_SUBMIT_DIR}/nlp/nano_gpt"; export PYTHONPATH="${PYTHONPATH:-}:${HOME}/batch_whitening"
python -u find_sync.py normalization_type=full_bw bw_renorm=True bw_renorm_stats=batch bw_group_size=384 batch_center_mode=running_mean bw_momentum=0.7 bw_warmup=none bw_fix_factor=0.8 bw_post_norm=ln_noaffine > ../study/logs/sync1.out 2>&1
echo "Job completed."
