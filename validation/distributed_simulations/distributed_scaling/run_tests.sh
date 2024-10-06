#!/bin/bash

#####
##### Disclaimer: this script can be used only on SLURM type systems.
##### For PBS systems `job_script.sh` needs to be corrected to use the `qsub` syntax
#####

# Before running, make sure that:
#
# 1) The julia environmental variables (defined in this file) point to the right path
#
# 2) the NGPUS_PER_NODE variable is correct (in this file)
#
# 3) If the system is equipped with nsys profiler it is possible to enable a trace with PROFILE_TRACE=1 (in this file)
#
# 4) Oceananigans is instantiated with the correct MPI build:
# 	(run these lines in a gpu node substituting modules and paths)
# 	$ module load my_cuda_module
# 	$ module load my_cuda_aware_mpi_module
# 	$ export JULIA_DEPOT_PATH="/path/to/depot"
# 	$ export JULIA="/path/to/julia"
#   $ $JULIA --check-bounds=no -e 'using Pkg; Pkg.add("MPIPreferences");'
# 	$ $JULIA --project --check-bounds=no -e 'using MPIPreferences; MPIPreferences.use_system_binaries()'
#   $ $JULIA --project --check-bounds=no -e 'using Pkg; Pkg.build("MPI")'
#   $ $JULIA --project --check-bounds=no -e 'using Pkg; Pkg.instantiate()'
#
# 5) correct modules are loaded in job_script.sh
#
# 6) SBATCH variables in the job_script.sh file are correct (check memory, time)
#
# 7) The system has at least max(RX) * max(RY) gpus
#
# 8) Choose if testing the hydrostatic or nonhydrostatic model (SIMULATION variable in this file)
#
# 9) Choose if measuring the weak or strong scaling (SCALING variable in this file)
#
# Finally -> $ ./run_tests.sh

# Julia specific enviromental variables
export JULIA_DEPOT_PATH="/path/to/depot"
export JULIA="/path/to/julia"

# PROFILE_TRACE=1 only if the system is equipped with nsys
export PROFILE_TRACE=0

# Number of gpus per node
export NGPUS_PER_NODE=4
	
# Choice between nonhydrostatic and hydrostatic
export SIMULATION=nonhydrostatic
# Choice between strong and weak
export SCALING=weak

for RX in 1 2 4 8 16 32 64; do
    for RY in 1 2 4 8 16 32 64; do
        
		export RX
        export RY

		if test $SIMULATION = "hydrostatic"; then 
			if test $SCALING = "weak"; then
				# Grid size for Weak scaling tests (Hydrostatic)
				export NX=$((1440 * RX))
				export NY=$((600 * RY))
				export NZ=100 
			else
				# Grid size for Strong scaling tests (Hydrostatic)
				export NX=1440
				export NY=600
				export NZ=100 
			fi
		else
			if test $SCALING = "weak"; then
				# Grid size for Weak scaling tests (Nonhydrostatic)
				export NX=$((512 * RX))
				export NY=$((512 * RY))
				export NZ=256 
			else
				# Grid size for Strong scaling tests (Nonhydrostatic)
				export NX=512
				export NY=512
				export NZ=256 
			fi
		fi
		
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

		# Use qsub on PBS systems!!!
		# qsub pbs_job_script.sh
    done
done
