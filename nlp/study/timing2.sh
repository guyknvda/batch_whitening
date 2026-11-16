#!/bin/bash
# Submit from the repo root: sbatch -J timing2 nlp/study/timing2.sh
#SBATCH --account=network_research_swarch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.out
#SBATCH --time=0:30:00
source "${HOME}/.bashrc"; source activate env_python39
cd "${SLURM_SUBMIT_DIR}"; export PYTHONPATH="${PYTHONPATH:-}:${HOME}/batch_whitening"
(cd nlp/nano_gpt && python -u bench_bw.py) > nlp/study/logs/bench-bw.out 2>&1
T="--max_iters=600 --eval_interval=10000 --wandb_log=False --log_conditioning=False"
BW="--normalization_type=full_bw --bw_renorm=True --bw_renorm_stats=batch --bw_group_size=384 --batch_center_mode=running_mean --bw_momentum=0.7 --bw_warmup=none --bw_fix_factor=0.8"
for cfg in "bw-nocache $BW --bw_renorm_cache=False" "bw-cache $BW" "bw-none $BW --bw_stat_grad=none"; do
  set -- $cfg; n=$1; shift
  python -u ./nlp/nano_gpt/train.py $T "$@" > nlp/study/logs/timing-$n.out 2>&1
done
echo "Job completed."
