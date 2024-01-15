# Run this script with
#
# $ mpiexec -n 4 julia --project distributed_geostrophic_adjustment.jl
#
# for example.
#
# You also probably should set
#
# $ export JULIA_NUM_THREADS=1

using MPI
using Oceananigans
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.DistributedComputations
using Oceananigans.Units: kilometers, meters
using Printf
using Logging

MPI.Init()

Logging.global_logger(OceananigansLogger())

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
Nranks = MPI.Comm_size(comm)

@info "Running on rank $rank of $Nranks..."

Lh = 100kilometers
Lz = 400meters
topology = (Bounded, Periodic, Bounded)

Nx = 80
# Distribute problem irregularly
# arch = Distributed(CPU(); partition = Partition(x = [10, 13, 18, 39]))
arch = Distributed(CPU(); partition=Partition(Nranks, 1, 1))

grid = RectilinearGrid(arch; topology,
                       size = (Nx, 3, 1),
                       x = (0, Lh),
                       y = (0, Lh),
                       z = (-Lz, 0))

@show grid

bottom(x, y) = x > 80kilometers && x < 90kilometers ? 100 : -500meters
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom))

model = HydrostaticFreeSurfaceModel(; grid,
                                    coriolis = FPlane(f=1e-4),
                                    free_surface = SplitExplicitFreeSurface(; substeps=10))

gaussian(x, L) = exp(-x^2 / 2L^2)

U = 0.1 # geostrophic velocity
L = Lh / 40 # gaussian width
x₀ = Lh / 4 # gaussian center
vᵍ(x, y, z) = -U * (x - x₀) / L * gaussian(x - x₀, L)

g = model.free_surface.gravitational_acceleration
η = model.free_surface.η
η₀ = model.coriolis.f * U * L / g # geostrophic free surface amplitude

ηᵍ(x) = η₀ * gaussian(x - x₀, L)
ηⁱ(x, y, z) = 2 * ηᵍ(x)

set!(model, v = vᵍ)
set!(model, η = ηⁱ)

gravity_wave_speed = sqrt(g * grid.Lz) # hydrostatic (shallow water) gravity wave speed
Δt = 2 * model.grid.Δxᶜᵃᵃ / gravity_wave_speed
simulation = Simulation(model; Δt, stop_iteration = 1000)

function progress_message(sim) 
    @info @sprintf("[%.2f%%], iteration: %d, time: %.3f, max|w|: %.2e",
    100 * sim.model.clock.time / sim.stop_time, sim.model.clock.iteration,
    sim.model.clock.time, maximum(abs, sim.model.velocities.u))
end

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(10))


outputs = merge(model.velocities, model.free_surface.η)
simulation.output_writers[:fields] = JLD2OutputWriter(model, outputs,
                                                      schedule = IterationInterval(1),
                                                      filename = "geostrophic_adjustment_rank$rank",
                                                      overwrite_existing = true)

run!(simulation)

