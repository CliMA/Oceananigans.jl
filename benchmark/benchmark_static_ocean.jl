using Printf
using TimerOutputs
using Oceananigans
using Oceananigans.Utils

include("benchmark_utils.jl")

#####
##### Benchmark setup and parameters
#####

const timer = TimerOutput()

Nt = 10  # Number of iterations to use for benchmarking time stepping.

threads = Threads.nthreads()

if threads == 1
    Ns = [(16, 16, 16), (32, 32, 32), (64, 64, 64), (128, 128, 128), (256, 256, 256)]
else
    Ns = [(256, 256, 256)]
end

   float_types = [Float32, Float64]
         archs = [CPU()]
@hascuda archs = [CPU(), GPU()]

#####
##### Run benchmarks
#####

for arch in archs, FT in float_types, N in Ns
    grid = RegularCartesianGrid(size=N, extent=(1, 1, 1))
    model = IncompressibleModel(architecture=arch, float_type=FT, grid=grid)

    time_step!(model, 1)  # precompile

    bname =  benchmark_name(N, "", arch, FT)
    @printf("Running static ocean benchmark: %s...\n", bname)
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

title = "Static ocean benchmarks"
title *= threads > 1 ? " ($threads threads)" : ""

print_timer(timer, title=title, sortby=:name)

println("\n\nCPU Float64 -> Float32 speedup:")
for N in Ns
    bn32 = benchmark_name(N, "", CPU(), Float32)
    bn64 = benchmark_name(N, "", CPU(), Float64)
    t32  = TimerOutputs.time(timer[bn32])
    t64  = TimerOutputs.time(timer[bn64])
    @printf("%s: %.3f\n", benchmark_name(N), t64/t32)
end

@hascuda begin
    println("\nGPU Float64 -> Float32 speedup:")
    for N in Ns
        bn32 = benchmark_name(N, "", GPU(), Float32)
        bn64 = benchmark_name(N, "", GPU(), Float64)
        t32  = TimerOutputs.time(timer[bn32])
        t64  = TimerOutputs.time(timer[bn64])
        @printf("%s: %.3f\n", benchmark_name(N), t64/t32)
    end

    println("\nCPU -> GPU speedup:")
    for N in Ns, ft in float_types
        bn_cpu = benchmark_name(N, "", CPU(), ft)
        bn_gpu = benchmark_name(N, "", GPU(), ft)
        t_cpu  = TimerOutputs.time(timer[bn_cpu])
        t_gpu  = TimerOutputs.time(timer[bn_gpu])
        @printf("%s: %.3f\n", benchmark_name(N, ft), t_cpu/t_gpu)
    end
end
