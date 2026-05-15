using CUDA
using Oceananigans.Architectures: GPU

const ROOT = normpath(joinpath(@__DIR__, ".."))
const TEST_PROJECT = joinpath(ROOT, "test")
const RUNTESTS = joinpath(ROOT, "test", "runtests.jl")
const MPI_PREFLIGHT_SCRIPT = "using MPI; MPI.Initialized() || MPI.Init(); println(\"MPI preflight OK\")"
const MPI_PREFLIGHT = `$(Base.julia_cmd()) --project=$TEST_PROJECT -e $MPI_PREFLIGHT_SCRIPT`
const LCC_GPU_TEST_FILES = (
    "test_lambert_conformal_conic_grid.jl",
    "test_grid_reconstruction.jl",
)

const DRY_RUN = "--dry-run" in ARGS

function validate_lcc_gpu_test_file(test_file)
    path = joinpath(@__DIR__, test_file)
    isfile(path) ||
        error("LambertConformalConicGrid GPU gate test file does not exist: $path")

    return nothing
end

function validate_lcc_gpu_environment(environment, test_file)
    environment["TEST_ARCHITECTURE"] == "GPU" ||
        error("LambertConformalConicGrid GPU gate requires TEST_ARCHITECTURE=GPU.")

    environment["TEST_FILE"] == test_file ||
        error("LambertConformalConicGrid GPU gate expected TEST_FILE=$test_file.")

    environment["MPI_TEST"] == "false" ||
        error("LambertConformalConicGrid GPU gate must run with MPI_TEST=false.")

    return nothing
end

function ensure_cuda_gpu_available()
    CUDA.functional() ||
        error("CUDA.functional() is false. Run this gate on CUDA GPU hardware.")

    GPU()
    return nothing
end

function ensure_mpi_available()
    run(MPI_PREFLIGHT)

    return nothing
end

function run_lcc_gpu_test_file(test_file; dry_run = false)
    environment = copy(ENV)
    environment["TEST_ARCHITECTURE"] = "GPU"
    environment["TEST_FILE"] = test_file
    # This disables distributed test branches. The standard test harness still
    # imports MPI and initializes MPI before reading this flag.
    environment["MPI_TEST"] = "false"

    validate_lcc_gpu_environment(environment, test_file)

    command = `$(Base.julia_cmd()) --project=$TEST_PROJECT $RUNTESTS`

    message = dry_run ? "Dry-running LambertConformalConicGrid GPU gate" :
                        "Running LambertConformalConicGrid GPU gate"

    test_file_environment = environment["TEST_FILE"]
    test_architecture = environment["TEST_ARCHITECTURE"]
    mpi_test = environment["MPI_TEST"]

    @info message test_file command test_file_environment test_architecture mpi_test

    if !dry_run
        run(setenv(command, environment))
    end

    return nothing
end

for test_file in LCC_GPU_TEST_FILES
    validate_lcc_gpu_test_file(test_file)
end

if !DRY_RUN
    ensure_cuda_gpu_available()
    ensure_mpi_available()
else
    @info "Dry-running LambertConformalConicGrid MPI preflight" command = MPI_PREFLIGHT
end

for test_file in LCC_GPU_TEST_FILES
    run_lcc_gpu_test_file(test_file; dry_run = DRY_RUN)
end
