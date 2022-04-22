push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using BenchmarkTools
using BSON
using Oceananigans
using Benchmarks

N = parse(Int, ARGS[1])
grid = RectilinearGrid(size=(N, N, N), extent=(1, 1, 1))
model = NonhydrostaticModel(architecture=CPU(), grid=grid)

time_step!(model, 1) # warmup

trial = @benchmark begin
    @sync_gpu time_step!($model, 1)
end samples=10

bson("multithreading_benchmark_$(Threads.nthreads()).bson", Dict(:trial => trial))
