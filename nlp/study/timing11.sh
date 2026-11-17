#!/bin/bash
# Fused post-BW LayerNorm (no gain or bias) vs unfused. Submit from the repo root: sbatch -J timing11 nlp/study/timing11.sh
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
(cd nlp/nano_gpt && python -u check_bw_fast.py) > nlp/study/logs/check-bw-fast-post.out 2>&1
python -u ./nlp/nano_gpt/train.py $T --normalization_type=layer_norm > nlp/study/logs/timing-ln-9.out 2>&1
python -u ./nlp/nano_gpt/train.py $T $BW --bw_fuse_post_norm=False > nlp/study/logs/timing-bw-ln0-unfused.out 2>&1
python -u ./nlp/nano_gpt/train.py $T $BW > nlp/study/logs/timing-bw-ln0-fused.out 2>&1
echo "Job completed."
