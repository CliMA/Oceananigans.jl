using Printf
using TimerOutputs
using Oceananigans
using Oceananigans.Utils

include("benchmark_utils.jl")

#####
##### Benchmark setup and parameters
#####

const timer = TimerOutput()

FT = Float64
Nt = 10  # Number of iterations to use for benchmarking time stepping.

         archs = [CPU()]        # Architectures to benchmark on.
@hascuda archs = [CPU(), GPU()] # Benchmark GPU on systems with CUDA-enabled GPUs.

#####
##### Utility functions for generating tracer lists
#####

function active_tracers(n)
    n == 0 && return []
    n == 1 && return [:b]
    n == 2 && return [:T, :S]
    throw(ArgumentError("Can't have more than 2 active tracers!"))
end

passive_tracers(n) = [Symbol("C" * string(n)) for n in 1:n]

tracer_list(na, np) = Tuple(vcat(active_tracers(na), passive_tracers(np)))

""" Number of active tracers to buoyancy """
function na2buoyancy(n)
    n == 0 && return nothing
    n == 1 && return BuoyancyTracer()
    n == 2 && return SeawaterBuoyancy()
    throw(ArgumentError("Can't have more than 2 active tracers!"))
end

#####
##### Run benchmarks
#####

# Each test case specifies (number of active tracers, number of passive tracers)
test_cases = [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0), (2, 3), (2, 5), (2, 10)]

for arch in archs, test_case in test_cases
    N = arch isa CPU ? (32, 32, 32) : (256, 256, 128)
    na, np = test_case
    tracers = tracer_list(na, np)

    grid = RegularCartesianGrid(size=N, extent=(1, 1, 1))
    model = IncompressibleModel(architecture=arch, float_type=FT, grid=grid,
                                buoyancy=na2buoyancy(na), tracers=tracers)

    time_step!(model, 1)  # precompile

    bname =  benchmark_name(N, "$na active + $(lpad(np, 2)) passive", arch, FT)
    @printf("Running benchmark: %s...\n", bname)
    for i in 1:Nt
        @timeit timer bname time_step!(model, 1)
    end
end

#####
##### Print benchmark results
#####

println()
println(oceananigans_versioninfo())
println(versioninfo_with_gpu())
print_timer(timer, title="Tracer benchmarks", sortby=:name)
println()
