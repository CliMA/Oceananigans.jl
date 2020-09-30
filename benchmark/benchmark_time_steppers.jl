using Printf
using TimerOutputs
using Oceananigans
using Oceananigans.Advection
using Oceananigans.Utils

include("benchmark_utils.jl")

#####
##### Benchmark setup and parameters
#####

const timer = TimerOutput()

Nt = 10  # Number of iterations to use for benchmarking time stepping.

   float_types = [Float64]       # Float types to benchmark.
         archs = [CPU()]         # Architectures to benchmark on.
@hascuda archs = [CPU(), GPU()]  # Benchmark GPU on systems with CUDA-enabled GPUs.

time_steppers = (:QuasiAdamsBashforth2, :RungeKutta3)

#####
##### Run benchmarks
#####

for arch in archs, FT in float_types, ts in time_steppers
    N = arch isa CPU ? 64 : 256
    
    grid = RegularCartesianGrid(FT, size=(N, N, N), extent=(1, 1, 1))
    model = IncompressibleModel(architecture=arch, float_type=FT, grid=grid, timestepper=ts)

    time_step!(model, 1)  # precompile

    bn =  benchmark_name((N, N, N), string(ts), arch, FT)
    @printf("Running benchmark: %s...\n", bn)
    for i in 1:Nt
        @timeit timer bn time_step!(model, 1)
    end
end

#####
##### Print benchmark results
#####

println()
println(oceananigans_versioninfo())
println(versioninfo_with_gpu())
print_timer(timer, title="Time stepper benchmarks", sortby=:name)
println()
