#!/bin/bash
# Warm-started Newton-Schulz cost. Submit from the repo root: sbatch -J timing5 nlp/study/timing5.sh
#SBATCH --account=network_research_swarch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.out
#SBATCH --time=0:30:00
source "${HOME}/.bashrc"; source activate env_python39
cd "${SLURM_SUBMIT_DIR}"; export PYTHONPATH="${PYTHONPATH:-}:${HOME}/batch_whitening"
(cd nlp/nano_gpt && python -u bench_bw.py "BW, cache, fused, NS warm 2") > nlp/study/logs/bench-bw-nswarm.out 2>&1
T="--max_iters=600 --eval_interval=10000 --wandb_log=False --log_conditioning=False"
BW="--normalization_type=full_bw --bw_renorm=True --bw_renorm_stats=batch --bw_group_size=384 --batch_center_mode=running_mean --bw_momentum=0.7 --bw_warmup=none --bw_fix_factor=0.8"
python -u ./nlp/nano_gpt/train.py $T --normalization_type=layer_norm > nlp/study/logs/timing-ln-3.out 2>&1
python -u ./nlp/nano_gpt/train.py $T $BW > nlp/study/logs/timing-bw-chol-3.out 2>&1
for k in 1 2 3; do
  python -u ./nlp/nano_gpt/train.py $T $BW --bw_whiten=ns_warm --bw_ns_iters=$k > nlp/study/logs/timing-bw-nswarm$k.out 2>&1
done
python -u ./nlp/nano_gpt/train.py $T $BW --bw_whiten=ns_warm --bw_ns_iters=2 --bw_ns_refresh=100 > nlp/study/logs/timing-bw-nswarm2-r100.out 2>&1
echo "Job completed."
