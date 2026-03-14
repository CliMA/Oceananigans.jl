#!/bin/bash
#PBS -N tripolar_tests
#PBS -l ncpus=8,mem=32GB,walltime=01:00:00
#PBS -l storage=gdata/y99
#PBS -l jobfs=4GB
#PBS -q express
#PBS -j oe
#PBS -l wd

# --- Environment setup ---
source /g/data/y99/bp3051/.julia/dev/Oceananigans/tmp_MPI_Gadi/defaults.sh

# Run from tmp_MPI_Gadi/ where parallel-compatible test files live
cd /g/data/y99/bp3051/.julia/dev/Oceananigans/tmp_MPI_Gadi
export JULIA_PROJECT=/g/data/y99/bp3051/.julia/dev/Oceananigans/tmp_MPI_Gadi

# --- Run distributed tripolar MPI tests ---
echo "=== Distributed TripolarGrid MPI tests ==="
echo "Start time: $(date)"
julia --project=. --check-bounds=yes -O0 test_mpi_tripolar.jl
echo "End time: $(date)"
echo "Exit status: $?"
