#!/usr/bin/env bash
#
# Local CPU mirror of .buildkite/distributed/pipeline.yml.
# Runs each distributed test group with 4 MPI ranks via mpiexec instead of srun.
#
# Usage:
#   ./run_distributed_pipeline_cpu.sh                  # run all groups
#   ./run_distributed_pipeline_cpu.sh distributed      # run a single group
#
set -eu

cd "$(dirname "$0")"

export JULIA_NUM_PRECOMPILE_TASKS=8
export MPI_TEST=true
export DATADEPS_ALWAYS_ACCEPT=true
export TEST_ARCHITECTURE=CPU

NRANKS="${NRANKS:-4}"
JULIA="${JULIA:-julia}"

# Pass groups on the command line, e.g.
#   ./run_distributed_pipeline_cpu.sh distributed distributed_solvers
# With no arguments, run all groups.
if [ "$#" -eq 0 ]; then
    set -- \
        distributed \
        distributed_solvers \
        distributed_hydrostatic_regression \
        distributed_hydrostatic_model \
        distributed_nonhydrostatic_regression \
        distributed_vertical_coordinate_1 \
        distributed_vertical_coordinate_2
fi

# Instantiate the test environment once up-front. We can NOT use `Pkg.test()`
# inside mpiexec: Pkg.test() spawns a child Julia process to run the tests, but
# only the parent is registered as an MPI rank with the PMI server, so the
# children fail in MPI_Init with `PMI_Get_appnum returned -1`. Instead, we
# launch test/runtests.jl directly under mpiexecjl so each rank IS the test
# process.
echo "--- Instantiating test environment"
"${JULIA}" -O0 --color=yes --project=test -e '
    using Pkg
    Pkg.develop(PackageSpec(path = pwd()))
    Pkg.instantiate()
    Pkg.precompile()
'

for group in "$@"; do
    echo "--- Running distributed test group: ${group} (CPU, ${NRANKS} ranks)"
    TEST_GROUP="${group}" \
        mpiexecjl -n "${NRANKS}" \
        "${JULIA}" -O0 --color=yes --project=test test/runtests.jl
done
