push!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Logging
using MPI
using JLD2
using BenchmarkTools

using Oceananigans
using Oceananigans.Distributed
using Benchmarks

Logging.global_logger(OceananigansLogger())

      comm = MPI.COMM_WORLD
local_rank = MPI.Comm_rank(comm)
         R = MPI.Comm_size(comm)

 #assigns one GPU per rank, could increase efficiency but must have enough GPUs
 #CUDA.device!(local_rank)

 Nx = parse(Int, ARGS[1])
 Ny = parse(Int, ARGS[2])
 Nz = parse(Int, ARGS[3])
 Rx = parse(Int, ARGS[4])
 Ry = parse(Int, ARGS[5])
 Rz = parse(Int, ARGS[6])

@assert Rx * Ry * Rz == R

@info "Setting up distributed nonhydrostatic model with N=($Nx, $Ny, $Nz) grid points and ranks=($Rx, $Ry, $Rz) on rank $local_rank..."

topo = (Periodic, Periodic, Periodic)
arch = MultiArch(CPU(), topology=topo, ranks=(Rx, Ry, Rz), communicator=MPI.COMM_WORLD)
distributed_grid = RectilinearGrid(arch, topology=topo, size=(Nx, Ny, Nz), extent=(1, 1, 1))
model = NonhydrostaticModel(grid=distributed_grid)

@info "Warming up distributed nonhydrostatic model on rank $local_rank..."

time_step!(model, 1) # warmup

@info "Benchmarking distributed nonhydrostatic model on rank $local_rank..."

MPI.Barrier(comm)

trial = @benchmark begin
    @sync_gpu time_step!($model, 1)
end samples=10 evals=1

MPI.Barrier(comm)

t_median = BenchmarkTools.prettytime(median(trial).time)
@info "Done benchmarking on rank $(local_rank). Median time: $t_median"

jldopen("distributed_nonhydrostatic_model_$(R)ranks_$local_rank.jld2", "w") do file
    file["trial"] = trial
end
