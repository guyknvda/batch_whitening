#!/bin/bash
# Small-matrix building blocks. Submit from the repo root: sbatch -J linalg1 nlp/study/linalg1.sh
#SBATCH --account=network_research_swarch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.out
#SBATCH --time=0:15:00
source "${HOME}/.bashrc"; source activate env_python39
cd "${SLURM_SUBMIT_DIR}/nlp/nano_gpt"; export PYTHONPATH="${PYTHONPATH:-}:${HOME}/batch_whitening"
python -u bench_linalg.py > ../study/logs/bench-linalg.out 2>&1
python -u profile_step.py normalization_type=full_bw bw_renorm=True bw_renorm_stats=batch bw_group_size=384 batch_center_mode=running_mean bw_momentum=0.7 bw_warmup=none bw_fix_factor=0.8 bw_post_norm=ln_noaffine bw_whiten=ns bw_ns_iters=10 2>&1 | grep -E 'wall|GPU span|idle|Error' >> ../study/logs/bench-linalg.out
echo "Job completed."
