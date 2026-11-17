#!/bin/bash
# Post-BW normalization without affine. Submit from the repo root: sbatch -J timing10 nlp/study/timing10.sh
#SBATCH --account=network_research_swarch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.out
#SBATCH --time=0:30:00
source "${HOME}/.bashrc"; source activate env_python39
cd "${SLURM_SUBMIT_DIR}"; export PYTHONPATH="${PYTHONPATH:-}:${HOME}/batch_whitening"
T="--max_iters=600 --eval_interval=10000 --wandb_log=False --log_conditioning=False"
BW="--normalization_type=full_bw --bw_renorm=True --bw_renorm_stats=batch --bw_group_size=384 --batch_center_mode=running_mean --bw_momentum=0.7 --bw_warmup=none --bw_fix_factor=0.8"
python -u ./nlp/nano_gpt/train.py $T --normalization_type=layer_norm > nlp/study/logs/timing-ln-8.out 2>&1
python -u ./nlp/nano_gpt/train.py $T $BW > nlp/study/logs/timing-bw-fast5-ln.out 2>&1
python -u ./nlp/nano_gpt/train.py $T $BW --bw_post_norm=ln_noaffine > nlp/study/logs/timing-bw-fast5-ln0.out 2>&1
python -u ./nlp/nano_gpt/train.py $T $BW --bw_post_norm=rms > nlp/study/logs/timing-bw-fast5-rms.out 2>&1
echo "Job completed."
