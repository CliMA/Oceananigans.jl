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
Nz = parse(Int, ARGS[3])

@info "Setting up distributed incompressible model with N=($Nx, $Ny, $Nz) grid points on $R rank(s)..."

topo = (Periodic, Periodic, Periodic)
distributed_grid = RegularRectilinearGrid(topology=topo, size=(Nx, Ny, Nz), extent=(1, 1, 1))
arch = MultiCPU(grid=distributed_grid, ranks=(1, R, 1))
model = DistributedIncompressibleModel(architecture=arch, grid=distributed_grid)

@info "Warming up distributed incompressible model..."

time_step!(model, 1) # warmup

@info "Benchmarking distributed incompressible model..."

trial = @benchmark begin
    @sync_gpu time_step!($model, 1)
    MPI.Barrier(comm)
end samples=10

@info "Rank $local_rank is done benchmarking!"

jldopen("strong_scaling_incompressible_model_$(R)_$local_rank.jld2", "w") do file
    file["trial"] = trial
end

