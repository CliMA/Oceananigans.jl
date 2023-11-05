#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=500GB
#SBATCH --time 24:00:00
#SBATCH -o output_RX${RX}_RY${RY}_NX${NX}_NY${NY}
#SBATCH -e error_RX${RX}_RY${RY}_NX${NX}_NY${NY}

## modules setup
# Upload modules: cuda and cuda-aware mpi
module purge all
module add spack
# Example:
# module add cuda/11.4
# module load openmpi/3.1.6-cuda-pmi-ucx-slurm-jhklron

# MPI specific exports (usually not needed)
# export OMPI_MCA_pml=^ucx
# export OMPI_MCA_osc=^ucx
# export OMPI_MCA_btl_openib_allow_ib=true

# Number of threads in SLURM mode
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:=1}

cat > launch.sh << EoF_s
#! /bin/sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
exec \$*
EoF_s
chmod +x launch.sh

# Add an NSYS trace only if the system has it
if test $PROFILE_TRACE == 1; then
   NSYS="nsys profile --trace=nvtx,cuda,mpi --output=report_RX${RX}_RY${RY}_NX${NX}_NY${NY}"
fi

$NSYS srun --mpi=pmi2 ./launch.sh $JULIA --check-bounds=no --project distributed_les.jl 