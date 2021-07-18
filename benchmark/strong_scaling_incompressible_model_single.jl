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

decomposition = ARGS[1]
Nx = parse(Int, ARGS[2])
Ny = parse(Int, ARGS[3])
Nz = parse(Int, ARGS[4])
Rx = parse(Int, ARGS[5])
Ry = parse(Int, ARGS[6])
Rz = parse(Int, ARGS[7])

@assert Rx * Ry * Rz == R

@info "Setting up distributed incompressible model with N=($Nx, $Ny, $Nz) grid points and ranks=($Rx, $Ry, $Rz) ($decomposition decomposition) on rank $local_rank..."

topo = (Periodic, Periodic, Periodic)
distributed_grid = RegularRectilinearGrid(topology=topo, size=(Nx, Ny, Nz), extent=(1, 1, 1))
arch = MultiCPU(grid=distributed_grid, ranks=(Rx, Ry, Rz))
model = DistributedNonhydrostaticModel(architecture=arch, grid=distributed_grid)

@info "Warming up distributed incompressible model on rank $local_rank..."

time_step!(model, 1) # warmup

@info "Benchmarking distributed incompressible model on rank $local_rank..."

trial = @benchmark begin
    @sync_gpu time_step!($model, 1)
end samples=10

t_median = BenchmarkTools.prettytime(median(trial).time)
@info "Done benchmarking on rank $(local_rank). Median time: $t_median"

jldopen("strong_scaling_incompressible_model_$(R)ranks_$(decomposition)_$local_rank.jld2", "w") do file
    file["trial"] = trial
end
