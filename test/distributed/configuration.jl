include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using MPI

function run_mpi_script(script; nranks = 1, threads = 1, cpus = nothing)
    mktempdir() do dir
        path = joinpath(dir, "script.jl")
        write(path, script)
        command = `$(mpiexec()) -n $nranks $(Base.julia_cmd()) -O0 --threads=$threads $path`
        return success(run(ignorestatus(isnothing(cpus) ? command : setcpuaffinity(command, cpus))))
    end
end

misconfigured_cpu_script = """
    using MPI
    MPI.Init(threadlevel = :serialized)
    using Oceananigans, Test, Logging

    cores_warning = Sys.islinux() ? ((:warn, r"fewer CPU cores than Julia threads"),) : ()
    @test_logs (:warn, r"single MPI rank") (:warn, r"MPI_THREAD_MULTIPLE") cores_warning... min_level=Logging.Warn Distributed(CPU())
"""

configured_cpu_script = """
    using MPI
    MPI.Init(threadlevel = :multiple)
    using Oceananigans, Test, Logging

    @test_logs min_level=Logging.Warn Distributed(CPU())
"""

shared_gpu_script = """
    ENV["JULIA_CUDA_MEMORY_POOL"] = "cuda"
    using MPI
    MPI.Init(threadlevel = :multiple)
    using Oceananigans, CUDA, Test, Logging

    expected_warnings = MPI.Comm_rank(MPI.COMM_WORLD) == 0 ? ((:warn, r"share one GPU"), (:warn, r"memory pool")) : ()
    @test_logs expected_warnings... match_mode=:any min_level=Logging.Warn Distributed(GPU(); devices = (0, 0))
"""

@testset "Distributed configuration warnings" begin
    # pins the job to one core; only Linux exposes CPU affinity
    @test run_mpi_script(misconfigured_cpu_script; threads = 2, cpus = Sys.islinux() ? [1] : nothing)
    @test run_mpi_script(configured_cpu_script; nranks = 2, threads = 2)
    CUDA.functional() && @test run_mpi_script(shared_gpu_script; nranks = 2)
end
