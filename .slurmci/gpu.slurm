#!/bin/bash

#SBATCH --time=1:00:00    # walltime
#SBATCH --nodes=1         # number of nodes
#SBATCH --mem-per-cpu=4G  # memory per CPU core
#SBATCH --gres=gpu:1

set -euo pipefail
set -x  # echo on

export PATH="${PATH}:${HOME}/julia-1.2/bin"
export JULIA_DEPOT_PATH="$(pwd)/.slurmdepot/${SLURM_JOB_ID}/${HOSTNAME}:$(pwd)/.slurmdepot/common"
export CUDA_PATH="/lib64"

module load cuda/9.1

julia --color=no --project -e 'using Pkg; pkg"instantiate"; pkg"precompile"'
julia --color=no --project $1
