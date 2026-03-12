#!/bin/bash
#PBS -N print_index
#PBS -l ncpus=4,mem=16GB,walltime=00:30:00
#PBS -l storage=gdata/y99
#PBS -l jobfs=4GB
#PBS -q express
#PBS -j oe
#PBS -l wd

# --- Environment setup ---

module load openmpi/5.0.8

export LD_LIBRARY_PATH=/apps/openmpi/5.0.8/lib
export JULIA_NUM_THREADS=1
export UCX_ERROR_SIGNALS="SIGILL,SIGBUS,SIGFPE"
export UCX_WARN_UNUSED_ENV_VARS=n
export MPITRAMPOLINE_LIB=$HOME/mpiwrapper/lib64/libmpiwrapper.so

LIBMOUNT_DIR=$(dirname "$(find "${JULIA_DEPOT_PATH:-$HOME/.julia}/artifacts" -name "libmount.so.1" -print -quit 2>/dev/null)" 2>/dev/null)
if [ -n "$LIBMOUNT_DIR" ]; then
    export LD_LIBRARY_PATH="${LIBMOUNT_DIR}:${LD_LIBRARY_PATH}"
fi

cd /g/data/y99/bp3051/.julia/dev/Oceananigans/tmp_MPI_Gadi
export JULIA_PROJECT=/g/data/y99/bp3051/.julia/dev/Oceananigans/tmp_MPI_Gadi

echo "=== CC location ==="
mpiexec -n 4 julia --project=. --check-bounds=yes -O0 print_index_CC.jl

echo ""
echo "=== FC location ==="
mpiexec -n 4 julia --project=. --check-bounds=yes -O0 print_index_FC.jl

echo ""
echo "=== CF location ==="
mpiexec -n 4 julia --project=. --check-bounds=yes -O0 print_index_CF.jl

echo ""
echo "=== FF location ==="
mpiexec -n 4 julia --project=. --check-bounds=yes -O0 print_index_FF.jl
