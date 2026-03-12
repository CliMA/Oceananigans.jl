#!/bin/bash
# Submit all testsets as parallel PBS jobs.
# Testsets 1 & 2: 4 configs each (2 folds × 2 partitions)
# Testsets 5 & 6: 3 configs each (slab, pencil, large-pencil)
# Testsets 3 & 4: single job each
#
# Usage: bash run_tripolar_tests_parallel.sh

DIR=/g/data/y99/bp3051/.julia/dev/Oceananigans/tmp_MPI_Gadi

# Testsets 1 & 2: split by config (1-4)
for ts in 1 2; do
    for cfg in 1 2 3 4; do
        qsub -P y99 -v "TRIPOLAR_TESTSET=$ts,TRIPOLAR_CONFIG=$cfg" \
             -N "tripolar_test_${ts}_cfg${cfg}" \
             $DIR/run_tripolar_tests.sh
    done
done

# Testsets 3 & 4: single job each
for ts in 3 4; do
    qsub -P y99 -v TRIPOLAR_TESTSET=$ts \
         -N "tripolar_test_$ts" \
         $DIR/run_tripolar_tests.sh
done

# Testsets 5 & 6: split by config (1-3)
for ts in 5 6; do
    for cfg in 1 2 3; do
        qsub -P y99 -v "TRIPOLAR_TESTSET=$ts,TRIPOLAR_CONFIG=$cfg" \
             -N "tripolar_test_${ts}_cfg${cfg}" \
             $DIR/run_tripolar_tests.sh
    done
done
