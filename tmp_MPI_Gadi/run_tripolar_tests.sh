#!/bin/bash
#PBS -N tripolar_tests
#PBS -l ncpus=8,mem=32GB,walltime=01:00:00
#PBS -l storage=gdata/y99
#PBS -l jobfs=4GB
#PBS -q express
#PBS -j oe
#PBS -l wd

# --- Environment setup (adapted from ACCESS-OM2_x_Oceananigans env_defaults.sh) ---

module load openmpi/5.0.8
module load cuda/12.9.0

export LD_LIBRARY_PATH=/apps/openmpi/5.0.8/lib
export JULIA_CUDA_USE_COMPAT=false
export JULIA_CUDA_MEMORY_POOL=none
export JULIA_NUM_THREADS=1
export UCX_ERROR_SIGNALS="SIGILL,SIGBUS,SIGFPE"
export UCX_WARN_UNUSED_ENV_VARS=n
export MPITRAMPOLINE_LIB=$HOME/mpiwrapper/lib64/libmpiwrapper.so

# Prepend JLL Libmount artifact to LD_LIBRARY_PATH
LIBMOUNT_DIR=$(dirname "$(find "${JULIA_DEPOT_PATH:-$HOME/.julia}/artifacts" -name "libmount.so.1" -print -quit 2>/dev/null)" 2>/dev/null)
if [ -n "$LIBMOUNT_DIR" ]; then
    export LD_LIBRARY_PATH="${LIBMOUNT_DIR}:${LD_LIBRARY_PATH}"
fi

# Run from tmp_MPI_Gadi/ where all test files and Project.toml live
cd /g/data/y99/bp3051/.julia/dev/Oceananigans/tmp_MPI_Gadi

# Ensure MPI subprocesses also use this environment
export JULIA_PROJECT=/g/data/y99/bp3051/.julia/dev/Oceananigans/tmp_MPI_Gadi

# --- Run distributed tripolar MPI tests ---
echo "=== Distributed TripolarGrid MPI tests ==="
julia --project=. --check-bounds=yes -O0 test_mpi_tripolar.jl
