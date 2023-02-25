push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Logging
using MPI
using JLD2
using BenchmarkTools

using Oceananigans
using Oceananigans.Distributed
using Benchmarks

Logging.global_logger(OceananigansLogger())

MPI.Init()

      comm = MPI.COMM_WORLD
local_rank = MPI.Comm_rank(comm)
         R = MPI.Comm_size(comm)

 #assigns one GPU per rank, could increase efficiency but must have enough GPUs
 #CUDA.device!(local_rank)

Nx = parse(Int, ARGS[1])
Ny = parse(Int, ARGS[2])
Rx = parse(Int, ARGS[3])
Ry = parse(Int, ARGS[4])

@assert Rx * Ry == R

@info "Setting up distributed shallow water model with N=($Nx, $Ny) grid points and ranks=($Rx, $Ry) on rank $local_rank..."

topo = (Periodic, Periodic, Flat)
arch = DistributedArch(CPU(), topology=topo, ranks=(Rx, Ry, 1), communicator=MPI.COMM_WORLD)
distributed_grid = RectilinearGrid(arch, topology=topo, size=(Nx, Ny), extent=(1, 1))
model = ShallowWaterModel(grid=distributed_grid, gravitational_acceleration=1.0)
set!(model, h=1)

@info "Warming up distributed shallow water model on rank $local_rank..."

time_step!(model, 1) # warmup

@info "Benchmarking distributed shallow water model on rank $local_rank..."

MPI.Barrier(comm)

trial = @benchmark begin
    @sync_gpu time_step!($model, 1)
end samples=10 evals=1

MPI.Barrier(comm)

t_median = BenchmarkTools.prettytime(median(trial).time)
@info "Done benchmarking on rank $(local_rank). Median time: $t_median"

jldopen("distributed_shallow_water_model_$(R)ranks_$local_rank.jld2", "w") do file
    file["trial"] = trial
end
