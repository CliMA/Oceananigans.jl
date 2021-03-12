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

Nx = parse(Int, ARGS[1])
Ny = parse(Int, ARGS[2])

@info "Setting up distributed shallow water model with N=($Nx, $Ny) grid points on $R rank(s)..."

topo = (Periodic, Periodic, Bounded)
distributed_grid = RegularRectilinearGrid(topology=topo, size=(Nx, Ny, 1), extent=(1, 1, 1))
arch = MultiCPU(grid=distributed_grid, ranks=(1, R, 1))
model = DistributedShallowWaterModel(architecture=arch, grid=distributed_grid, gravitational_acceleration=1.0)
set!(model, h=model.grid.Lz)

@info "Warming up distributed shallow water model..."

time_step!(model, 1) # warmup

@info "Benchmarking distributed shallow water model..."

trial = @benchmark begin
    @sync_gpu time_step!($model, 1)
    MPI.Barrier(comm)
end samples=10

@info "Rank $local_rank is done benchmarking!"

jldopen("strong_scaling_shallow_water_model_$(R)_$local_rank.jld2", "w") do file
    file["trial"] = trial
end

