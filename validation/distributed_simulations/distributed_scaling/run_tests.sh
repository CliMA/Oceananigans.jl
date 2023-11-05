#!/bin/bash

# Before running, make sure that:
#
# 1) The julia environmental variables (defined in this file) point to the right path
#
# 2) the NGPUS_PER_NODE variable is correct (in this file)
#
# 3) If the system is equipped with nsys profiler it is possible to enable a trace with PROFILE_TRACE=1 (in this file)
#
# 4) Oceananigans is instantiated with the correct MPI build:
# 	(run these lines in a gpu node)
# 	$ module load my_cuda_module
# 	$ module load my_cuda_aware_mpi_module
# 	$ export JULIA_DEPOT_PATH="/path/to/depot"
# 	$ export JULIA="/path/to/julia"
#   $ $JULIA --check-bounds=no -e 'using Pkg; Pkg.add("MPIPreferences");'
# 	$ $JULIA --project --check-bounds=no -e 'using MPIPreferences; MPIPreferences.use_system_binaries()'
#   $ $JULIA --project --check-bounds=no -e 'using Pkg; Pkg.build("MPI")'
#   $ $JULIA --project --check-bounds=no -e 'using Pkg; Pkg.instantiate("")'
#
# 5) correct modules are loaded in job_script.sh
#
# 6) SBATCH variables are in job_script.sh are correct (check memory, time)
#
# 7) The system has at least max(RX) * max(RY) gpus
#
# 8) Swith to the Strong scaling grid to test strong scaling (in this file)
#
#
# Finally -> $ ./run_tests.sh

# Julia specific enviromental variables
export JULIA_DEPOT_PATH="/path/to/depot"
export JULIA="/path/to/julia"

# PROFILE_TRACE=1 only if the system is equipped with nsys
export PROFILE_TRACE=0

# Number of gpus per node
export NGPUS_PER_NODE=4
	
for RX in 1 2 4 8 16 32 64; do
    for RY in 1 2 4 8 16 32 64; do
        
		export RX
        export RY

		# Grid size for Weak scaling tests
		export NX=$((512 * RX))
		export NY=$((512 * RY))
		export NZ=256 

		# Grid size for Strong scaling tests
		# export NX=512
		# export NY=512
		# export NZ=256 

		RANKS=$((RX * RY))

		export NNODES=$((RANKS / NGPUS_PER_NODE))
		export NTASKS=$NGPUS_PER_NODE

		echo ""
		echo "(RX, RY) = $RX, $RY"
		echo "(NX, NY) = $NX, $NY"
		echo "(NNODES, NTASKS) = $NNODES, $NTASKS"

		# ====================================================== #
		# ================== RUN SCALING TEST ================== #
		# ====================================================== #

		sbatch -N ${NNODES} --gres=gpu:${NTASKS} --ntasks-per-node=${NTASKS} job_script.sh
    done
done
