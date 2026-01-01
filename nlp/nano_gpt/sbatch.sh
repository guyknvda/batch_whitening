#!/bin/bash

# Example: sbatch sbatch.sh
# To change to 2 nodes: (1) -w ..., (2) --nodes=2, (3) -node_hostnames ..., (4) commented srun -N1 -n1 -w ... commands

#SBATCH --account=network_research_swarch
#SBATCH --nodes=1
# TODO: if specific nodes are allocated use as follows: -w cw-dfw-h100-001-014-013,cw-dfw-h100-001-046-026,cw-dfw-h100-001-096-012,cw-dfw-h100-001-366-026
#SBATCH --output=%x_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=4:00:00       # 4 hours

# Activate conda environment
source ${HOME}/.bashrc
source activate env_python39

export PYTHONPATH=$PYTHONPATH:${HOME}/batch_whitening
export WANDB_API_KEY=75f383306c29ff0d7c26dee55d76d5adbd67b07c

# Get the list of nodes allocated to the job and assign nodes to variables
NODES=($(scontrol show hostname))
NODE1=${NODES[0]}
echo "Node 1: $NODE1"

# Run each job on a different node
echo "Starting srun commands..."
srun -w ${NODE1} --unbuffered torchrun --standalone --nproc_per_node=8 ./nlp/nano_gpt/train.py &
SRUNPID1="$!"

echo "Waiting for all srun sub processes to complete..."
wait $SRUNPID1
echo "Jobs have completed."
