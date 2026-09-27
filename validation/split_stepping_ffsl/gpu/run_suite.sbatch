#!/bin/bash
#SBATCH -A prio_ey239
#SBATCH --job-name=bgc-split-ffsl-benchmark
#SBATCH --partition=priority_gpu
#SBATCH --gpus=h200:1
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=%x-%j.out
#SBATCH --mail-type=ALL

# Full GPU benchmark suite for split tracer stepping + FluxFormSemiLagrangian on a global tripolar BGC model.
# Submit from the repository root, on branch jsw/split-stepping-ffsl:
#
#   sbatch validation/split_stepping_ffsl/gpu/run_suite.sbatch
#
# Optional environment variables (pass with `sbatch --export=ALL,RUN_QUARTER_DEGREE=false ...`):
#   RUN_TESTS=true|false           GPU correctness tests before benchmarking (default true)
#   RUN_QUARTER_DEGREE=true|false  1/4° run with 24 BGC tracers after the 1° runs (default true)
#   SIMULATED_DAYS=30              simulated days per configuration

set -uo pipefail

module purge
module load Julia

repository=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$repository"

run_tests=${RUN_TESTS:-true}
run_quarter_degree=${RUN_QUARTER_DEGREE:-true}
simulated_days=${SIMULATED_DAYS:-30}
benchmark_environment="$repository/validation/split_stepping_ffsl/gpu/environment"
results="$repository/validation/split_stepping_ffsl/gpu/results/${SLURM_JOB_ID:-local}"
benchmark="$repository/validation/split_stepping_ffsl/gpu/bgc_speedup.jl"
mkdir -p "$results"

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TEST_ARCHITECTURE=GPU

declare -a summary

run_step() {
    local name=$1
    shift
    echo "==== $(date '+%F %T') start: $name"
    "$@" > "$results/$name.log" 2>&1
    local status=$?
    echo "==== $(date '+%F %T') end: $name (exit $status), log: $results/$name.log"
    summary+=("$name: exit $status")
    return $status
}

echo "Repository: $repository, branch: $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
echo "Results: $results"
nvidia-smi

# 1. Environments: the repository project (for the tests) and a benchmark environment with CUDA
run_step setup julia --project=. -e '
    using Pkg
    Pkg.instantiate()
    Pkg.activate(ARGS[1])
    Pkg.develop(path = ".")
    Pkg.add(["CUDA", "Adapt"])
    Pkg.instantiate()
    Pkg.precompile()
    using CUDA
    CUDA.versioninfo()
    CUDA.functional() || error("CUDA is not functional on this node")
' "$benchmark_environment" || { echo "Setup failed; see $results/setup.log"; exit 1; }

# 2. GPU correctness: the new tests must pass before the timings mean anything
if [ "$run_tests" = true ]; then
    run_step gpu_tests julia --project=. -e '
        using Pkg
        Pkg.test("Oceananigans"; test_args = ["--jobs=1",
            "vertical_coordinate/split_ffsl_conservation",
            "vertical_coordinate/split_tracer_conservation",
            "vertical_coordinate/ffsl_conservation",
            "time_stepping/split_tracer_stepping",
            "time_stepping/sinking_implicit_advection",
            "unit/flux_form_semi_lagrangian",
            "hydrostatic_free_surface/flux_form_semi_lagrangian_models"])
    ' || echo "GPU tests failed: benchmarks still run, but check $results/gpu_tests.log before trusting them"
fi

# 3. BGC speed-up on the 1° tripolar model, with 4 and 24 BGC tracers
for tracers in 4 24; do
    run_step "bgc_speedup_1deg_${tracers}bgc" \
        julia --project="$benchmark_environment" "$benchmark" 1 "$tracers" "$simulated_days" "$results"
done

# 4. Resolution scaling: 1/4° with 24 BGC tracers
if [ "$run_quarter_degree" = true ]; then
    run_step bgc_speedup_0p25deg_24bgc \
        julia --project="$benchmark_environment" "$benchmark" 0.25 24 "$simulated_days" "$results"
fi

echo
echo "==== Summary"
printf '%s\n' "${summary[@]}"
echo
for table in "$results"/bgc_speedup_*.md; do
    [ -f "$table" ] && cat "$table" && echo
done
