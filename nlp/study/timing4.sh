#!/bin/bash
# Newton-Schulz whitening cost. Submit from the repo root: sbatch -J timing4 nlp/study/timing4.sh
#SBATCH --account=network_research_swarch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.out
#SBATCH --time=0:30:00
source "${HOME}/.bashrc"; source activate env_python39
cd "${SLURM_SUBMIT_DIR}"; export PYTHONPATH="${PYTHONPATH:-}:${HOME}/batch_whitening"
# bench already in nlp/study/logs/bench-bw-ns.out
T="--max_iters=600 --eval_interval=10000 --wandb_log=False --log_conditioning=False"
BW="--normalization_type=full_bw --bw_renorm=True --bw_renorm_stats=batch --bw_group_size=384 --batch_center_mode=running_mean --bw_momentum=0.7 --bw_warmup=none --bw_fix_factor=0.8"


for k in 10; do
  python -u ./nlp/nano_gpt/train.py $T $BW --bw_whiten=ns --bw_ns_iters=$k > nlp/study/logs/timing-bw-ns$k.out 2>&1
done
echo "Job completed."
