#!/bin/bash

for RX in 1 2 4 8 16 32 64; do
    for RY in 1 2 4 8 16 32 64; do
        export $RX
        export $RY

		# ====================================================== #
		# ================ USER SPECIFIED INPUTS =============== #
		# ====================================================== #

		# PROFILE_TRACE=1 only if the system is equipped with nsys
		export PROFILE_TRACE=0

		# Grid size for Weak scaling tests
		export NX=$((512 * RX))
		export NY=$((512 * RY))
		export NZ=256 

		# Grid size for Strong scaling tests
		# export NX=512
		# export NY=512
		# export NZ=256 

		RANKS=$((RX * RY))

		export NGPUS_PER_NODE=4
		export NNODES=$((RANKS / NGPUS_PER_NODE))
		export NTASKS=$((RX * RY / NNODES))

		echo ""
		echo "(RX, RY) = $RX, $RY"
		echo "(NX, NY) = $NX, $NY"
		echo "(NNODES, NTASKS) = $NNODES, $NTASKS"

		# Julia specific enviromental variables
		export COMMON="/path/to/common/folder"
		export JULIA_DEPOT_PATH="${COMMON}/depot"
		export JULIA_CUDA_MEMORY_POOL=none
		export JULIA="${COMMON}/path/to/julia"

		# Profile specific variable
		export JULIA_NVTX_CALLBACKS=gc

		# ====================================================== #
		# ================== RUN SCALING TEST ================== #
		# ====================================================== #

		sbatch -N ${NNODES} --gres=gpu:${NTASKS} --ntasks-per-node=${NTASKS} job_script.sh
    done
done
