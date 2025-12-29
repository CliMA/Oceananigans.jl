# Distributed geostrophic adjustment validation
#
# Run with:
#
#   mpiexec -n 4 julia --project distributed_geostrophic_adjustment.jl
#

using MPI
using Oceananigans
using Oceananigans.DistributedComputations
using Oceananigans.DistributedComputations: Sizes
using Oceananigans.Units: kilometers, meters
using Printf

topology = (Bounded, Periodic, Bounded)
partition = Partition(x = Sizes(10, 13, 18, 39))

arch = Distributed(CPU(); partition)
rank = MPI.Comm_rank(arch.communicator)

# Distribute problem irregularly
Nx = 80
Lh = 100kilometers
Lz = 400meters

grid = RectilinearGrid(arch,
                       size = (Nx, 3, 2),
                       x = (0, Lh),
                       y = (0, Lh),
                       z = (-Lz, 0),
                       topology = topology)

@show rank, grid

bottom(x, y) = x > 80kilometers && x < 90kilometers ? 100 : -500meters
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom); active_cells_map = true)

coriolis = FPlane(f=1e-4)

model = HydrostaticFreeSurfaceModel(grid; coriolis,
                                    timestepper = :SplitRungeKutta3,
                                    free_surface = SplitExplicitFreeSurface(grid; substeps=10))

# Initial conditions: Gaussian geostrophic jet
gaussian(x, L) = exp(-x^2 / 2L^2)

U = 0.1 # geostrophic velocity
L = Lh / 40 # gaussian width
x₀ = Lh / 4 # gaussian center
vᵍ(x, y, z) = -U * (x - x₀) / L * gaussian(x - x₀, L)

g = model.free_surface.gravitational_acceleration
η₀ = coriolis.f * U * L / g # geostrophic free surface amplitude

ηᵍ(x) = η₀ * gaussian(x - x₀, L)
ηⁱ(x, y, z) = 2 * ηᵍ(x)

set!(model, v = vᵍ)
set!(model, η = ηⁱ)

gravity_wave_speed = sqrt(g * grid.Lz)
Δt = 2 * model.grid.Δxᶜᵃᵃ / gravity_wave_speed

simulation = Simulation(model; Δt, stop_iteration = 100)

function progress_message(sim)
    @info @sprintf("[%.2f%%], iteration: %d, time: %.3f, max|u|: %.2e",
                   100 * sim.model.clock.iteration / sim.stop_iteration,
                   sim.model.clock.iteration,
                   sim.model.clock.time,
                   maximum(abs, sim.model.velocities.u))
end

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(10))

run!(simulation)

@info "Simulation completed on rank $rank"
