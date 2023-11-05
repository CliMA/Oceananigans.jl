#!/bin/bash

# ====================================================== #
# ================ USER SPECIFIED INPUTS =============== #
# ====================================================== #

export PROFILE=1

# Grid size
export NX=$((512 * RX))
export NY=$((512 * RY))
export NZ=256 

TOTCORES=$((RX * RY))

if [ $TOTCORES -le 4 ]; then
	NODES=1	
else
	NODES=2
fi

export NNODES=$NODES

export NTASKS=$((RX * RY / NNODES))

echo ""
echo "(RX, RY) = $RX, $RY"
echo "(NX, NY) = $NX, $NY"
echo "(NNODES, NTASKS) = $NNODES, $NTASKS"

# ====================================================== #
# ================== RUN SCALING TEST ================== #
# ====================================================== #

sbatch -N ${NNODES} --gres=gpu:${NTASKS} --ntasks-per-node=${NTASKS} job_script.sh