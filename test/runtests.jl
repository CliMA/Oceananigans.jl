using ParallelTestRunner
using ParallelTestRunner: find_tests, parse_args, filter_tests!, runtests, addworker, default_njobs
using Oceananigans
using CUDA

const SETUP = joinpath(@__DIR__, "setup")
const on_gpu = get(ENV, "TEST_ARCHITECTURE", "CPU") == "GPU"
mpi_test = get(ENV, "MPI_TEST", "false") == "true"

# TEST_GROUP=a,b is a comma-separated list of test-name prefixes, equivalent to positional test_args.
const group = get(ENV, "TEST_GROUP", "")

if mpi_test
    # The distributed pipeline launches this file as 4 MPI ranks (`srun julia ... Pkg.test()`), so the
    # ranks themselves must execute the tests, in lockstep, without worker processes.
    include(joinpath(SETUP, "run_mpi_tests.jl"))
else
    args_vector = copy(ARGS)
    isempty(group) || append!(args_vector, split(group, ","))
    args = parse_args(args_vector)

    testsuite = find_tests(@__DIR__)

    # setup/ holds shared preludes, manual/ tests need credentials, mpi/ tests need 4 MPI ranks (see above).
    for prefix in ("setup/", "manual/", "mpi/")
        filter!(((name, _),) -> !startswith(name, prefix), testsuite)
    end
    delete!(testsuite, "sharding/tripolar") # TripolarGrid + ImmersedBoundaryGrid cause Reactant MLIR errors

    # Some tests don't run on GPU, always remove them.
    if on_gpu
        for prefix in ("enzyme/", "conservative_regridding/", "unit/operators", "unit/quality_assurance",
                       "unit/schedules", "unit/utils", "unit/weno_smoothness_reference", "unit/weno_smoothness")
            filter!(((name, _),) -> !startswith(name, prefix), testsuite)
        end
    end

    if filter_tests!(testsuite, args)
        # No explicit selection: skip suites that need extra hardware, a working MPI launcher, or a
        # different Julia version.
        for prefix in ("distributed/", "enzyme/", "reactant/", "sharding/", "metal/", "amdgpu/",
                       "oneapi/", "makie/", "convergence/")
            filter!(((name, _),) -> !startswith(name, prefix), testsuite)
        end
    end

    # init/init warms the depot and must finish before anything else; the distributed tests each spawn
    # their own 4-rank job, so they never overlap. Serial tests run before the parallel batch.
    serial = filter(
        name -> name == "init/init" || startswith(name, "distributed/") || startswith(name, "sharding/"),
        collect(keys(testsuite)))

    # Download reference data once, in this process, so workers only hit the DataDeps cache.
    needs_data(name) = startswith(name, "regression/") || name == "unit/grids" || startswith(name, "multi_region/cubed_sphere")
    args.list === nothing && any(needs_data, keys(testsuite)) && include(joinpath(SETUP, "data_dependencies.jl"))

    # Tests that mutate process-global state (loggers, Enzyme and Reactant flags, the active project,
    # the default float type) get a throw-away worker.
    dedicated_prefixes = ("enzyme/", "sharding/", "convergence/", "metal/", "oneapi/")
    function test_worker(name)
        if any(prefix -> startswith(name, prefix), dedicated_prefixes)
            addworker()
        elseif startswith(name, "memory_allocation/")
            addworker(; exeflags=["--check-bounds=auto"])
        else
            nothing
        end
    end

    function gpu_free_memory()
        using_cuda = on_gpu && CUDA.functional()
        using_cuda || return typemax(Int)
        device = first(CUDA.devices())
        if CUDA.has_nvml()
            mig = CUDA.uuid(device) != CUDA.parent_uuid(device)
            return Int(CUDA.NVML.memory_info(CUDA.NVML.Device(CUDA.uuid(device); mig)).free)
        else
            return CUDA.device!(device) do
                Int(CUDA.free_memory())
            end
        end
    end

    if args.jobs === nothing
        cpu_memory_per_worker = 4 * 2^30
        gpu_memory_per_worker = 3 * 2^30
        available_gpu_memory = gpu_free_memory()

        if CUDA.functional()
            println("Available CUDA GPU memory: ", Base.format_bytes(available_gpu_memory))
        end

        jobs = default_njobs()
        jobs = min(jobs, max(1, Int(Sys.free_memory()) ÷ cpu_memory_per_worker))
        jobs = min(jobs, max(1, available_gpu_memory ÷ gpu_memory_per_worker))
        args = ParallelTestRunner.ParsedArgs(Some(jobs), args.verbose, args.quickfail, args.list,
                                             args.custom, args.positionals)
    end
    jobs = something(args.jobs)

    # A worker with Oceananigans and CUDA loaded already uses a few GB, so the runner's default
    # threshold would recycle it after nearly every test.
    max_worker_rss = max(ParallelTestRunner.get_max_worker_rss(),
                         min(Int(Sys.total_memory()) ÷ (2jobs), 10 * 2^30))

    runtests(Oceananigans, args;
             testsuite,
             test_worker,
             serial,
             max_worker_rss,
             recycle_on_failure = true,
             history_key = on_gpu ? "gpu" : nothing,
             )
end
