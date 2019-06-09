#!/bin/bash

#SBATCH --time=1:00:00    # walltime
#SBATCH --nodes=1         # number of nodes
#SBATCH --mem-per-cpu=4G  # memory per CPU core

set -euo pipefail
set -x  # echo on

export PATH="${PATH}:${HOME}/julia-1.2/bin"
export JULIA_DEPOT_PATH="$(pwd)/.slurmdepot/common"

julia --color=no --project -e 'using Pkg; pkg"instantiate"; pkg"build"; pkg"precompile"'
