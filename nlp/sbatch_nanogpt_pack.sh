#!/bin/bash

# Example (from the repo root): sbatch -J pack1 nlp/sbatch_nanogpt_pack.sh nlp/study/pack1.txt
# Runs several nanoGPT jobs concurrently on one GPU (a nanoGPT run uses ~12% of an H100).
# Each line of the job file is "<run_name> <train.py overrides...>"; blank lines and # comments are skipped.
# Logs go to nlp/study/logs/<run_name>.out. MAX_PAR (default 4) runs share the GPU at a time.

#SBATCH --account=network_research_swarch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --output=%x_%j.out
#SBATCH --time=2:00:00

set -uo pipefail

source "${HOME}/.bashrc"
source activate env_python39

cd "${SLURM_SUBMIT_DIR:-${HOME}/batch_whitening}"
export PYTHONPATH="${PYTHONPATH:-}:${HOME}/batch_whitening"
export WANDB_DIR="/tmp/wandb_${SLURM_JOB_ID:-local}"
mkdir -p "$WANDB_DIR" nlp/study/logs

echo "Job ID: ${SLURM_JOB_ID:-unknown}  Node: ${SLURMD_NODENAME:-unknown}  File: $1"

# Without MPS the processes only time-slice the GPU (no throughput gain); MPS
# lets their small kernels run concurrently. Fall back to plain sharing if it fails.
export CUDA_MPS_PIPE_DIRECTORY="/tmp/mps_${SLURM_JOB_ID:-local}/pipe"
export CUDA_MPS_LOG_DIRECTORY="/tmp/mps_${SLURM_JOB_ID:-local}/log"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
MPS=0
if command -v nvidia-cuda-mps-control >/dev/null && nvidia-cuda-mps-control -d; then
    if python -c "import torch; torch.zeros(1, device='cuda')"; then
        MPS=1
    else
        echo quit | nvidia-cuda-mps-control
    fi
fi
[[ $MPS == 0 ]] && unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
echo "MPS: $MPS"

MAX_PAR="${MAX_PAR:-4}"
while read -r name args; do
    [[ -z "$name" || "$name" == \#* ]] && continue
    while (( $(jobs -rp | wc -l) >= MAX_PAR )); do sleep 5; done
    echo "start $name: $args"
    python -u ./nlp/nano_gpt/train.py --wandb_run_name="$name" --out_dir="/tmp/out-$name" $args \
        > "nlp/study/logs/$name.out" 2>&1 &
done < "$1"
wait
[[ $MPS == 1 ]] && echo quit | nvidia-cuda-mps-control
echo "Job completed."
