# Distributed shallow water turbulence validation
#
# Run with:
#
#   mpiexec -n 4 julia --project distributed_shallow_water_turbulence.jl
#

using MPI
using Oceananigans
using Oceananigans.DistributedComputations
using Statistics
using Printf

ranks = (2, 2, 1)
topology = (Periodic, Periodic, Flat)

arch = Distributed(CPU(), partition = Partition(ranks...))
rank = MPI.Comm_rank(arch.communicator)

Nx = 64
Ny = 64

grid = RectilinearGrid(arch,
                       topology = topology,
                       size = (Nx ÷ ranks[1], Ny ÷ ranks[2]),
                       extent = (4π, 4π),
                       halo = (3, 3))

model = ShallowWaterModel(grid;
                          timestepper = :RungeKutta3,
                          momentum_advection = UpwindBiased(order = 5),
                          gravitational_acceleration = 1)

set!(model, h = 1)

uh₀ = rand(size(model.grid)...)
uh₀ .-= mean(uh₀)
set!(model, uh = uh₀, vh = uh₀)

simulation = Simulation(model, Δt = 1e-3, stop_iteration = 100)

function progress(sim)
    @info @sprintf("Rank %d: iteration %d, time: %.4f",
                   rank, sim.model.clock.iteration, sim.model.clock.time)
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

MPI.Barrier(arch.communicator)

run!(simulation)

@info "Simulation completed on rank $rank"

MPI.Barrier(arch.communicator)
