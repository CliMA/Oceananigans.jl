push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Logging
using JLD2
using BenchmarkTools
using Benchmarks

using Oceananigans
using Oceananigans.Models
using CUDA

Logging.global_logger(OceananigansLogger())

Nx = parse(Int, ARGS[1])
Ny = parse(Int, ARGS[2])
Nz = parse(Int, ARGS[3])

T = Threads.nthreads()

@info "Setting up threaded serial nonhydrostatic model with N=($Nx, $Ny, $Nz) grid points and $T threads..."

topo = (Periodic, Periodic, Periodic)
grid = RegularRectilinearGrid(topology=topo, size=(Nx, Ny, Nz), extent=(1, 1, 1))
model = NonhydrostaticModel(architecture=CPU(), grid=grid)

@info "Warming up serial nonhydrostatic model..."

time_step!(model, 1) # warmup

@info "Benchmarking serial nonhydrostatic model..."

trial = @benchmark begin
    @sync_gpu time_step!($model, 1)
    #CUDA.@sync blocking=true time_step!($model, 1)
end samples=10 evals=1

t_median = BenchmarkTools.prettytime(median(trial).time)
@info "Done benchmarking. Median time: $t_median"

jldopen("distributed_nonhydrostatic_model_threads$T.jld2", "w") do file
    file["trial"] = trial
end
