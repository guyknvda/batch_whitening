#!/bin/bash
# Fast BW path (bw_fast). Submit from the repo root: sbatch -J timing6 nlp/study/timing6.sh
#SBATCH --account=network_research_swarch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.out
#SBATCH --time=0:30:00
source "${HOME}/.bashrc"; source activate env_python39
cd "${SLURM_SUBMIT_DIR}"; export PYTHONPATH="${PYTHONPATH:-}:${HOME}/batch_whitening"
(cd nlp/nano_gpt && python -u bench_bw.py "BW fast, Cholesky") > nlp/study/logs/bench-bw-fast.out 2>&1
T="--max_iters=600 --eval_interval=10000 --wandb_log=False --log_conditioning=False"
BW="--normalization_type=full_bw --bw_renorm=True --bw_renorm_stats=batch --bw_group_size=384 --batch_center_mode=running_mean --bw_momentum=0.7 --bw_warmup=none --bw_fix_factor=0.8"
python -u ./nlp/nano_gpt/train.py $T --normalization_type=layer_norm > nlp/study/logs/timing-ln-4.out 2>&1
python -u ./nlp/nano_gpt/train.py $T $BW > nlp/study/logs/timing-bw-chol-4.out 2>&1
python -u ./nlp/nano_gpt/train.py $T $BW --bw_fast=True > nlp/study/logs/timing-bw-fast-chol.out 2>&1
python -u ./nlp/nano_gpt/train.py $T $BW --bw_fast=True --bw_whiten=ns --bw_ns_iters=10 > nlp/study/logs/timing-bw-fast-ns10.out 2>&1
echo "Job completed."
