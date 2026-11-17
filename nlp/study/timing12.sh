#!/bin/bash
# GPU check and synchronized step timing of bw_whiten=ns_chol. Submit from the repo root: sbatch -J timing12 nlp/study/timing12.sh
#SBATCH --account=network_research_swarch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.out
#SBATCH --time=0:30:00
source "${HOME}/.bashrc"; source activate env_python39
cd "${SLURM_SUBMIT_DIR}/nlp/nano_gpt"; export PYTHONPATH="${PYTHONPATH:-}:${HOME}/batch_whitening"
python -u check_ns_chol.py cuda > ../study/logs/check-ns-chol-defer.out 2>&1
BW="normalization_type=full_bw bw_renorm=True bw_renorm_stats=batch bw_group_size=384 batch_center_mode=running_mean bw_momentum=0.7 bw_warmup=none bw_fix_factor=0.8"
for extra in "bw_whiten=ns_chol bw_ns_iters=10" "bw_whiten=ns_chol bw_ns_iters=10 bw_where=0,2,4"; do
  echo "== $extra"; python -u profile_step.py $BW bw_post_norm=ln_noaffine $extra
done 2>&1 | grep -E "==|wall|Error" > ../study/logs/timing-nschol-defer.out
echo "Job completed."
