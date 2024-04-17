using Oceananigans
using Oceananigans.MultiRegion
using Oceananigans.Utils
using Oceananigans.BuoyancyModels: g_Earth
using Oceananigans.Grids: topology, architecture
using Oceananigans.Units: kilometers, meters
using Printf
using JLD2

topo = (Bounded, Periodic, Bounded)

arch = CPU()

# Distribute problem irregularly
Nx = 80
Lh = 100kilometers
Lz = 400meters

grid = RectilinearGrid(arch,
                       size = (Nx, 3, 1),
                       x = (0, Lh),
                       y = (0, Lh),
                       z = (-Lz, 0),
                       halo = (2, 2, 2),
                       topology = topo)


grid = MultiRegionGrid(grid, partition = XPartition(4))

bottom(x, y) = x > 80kilometers && x < 90kilometers ? 100 : -500meters
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom))

free_surface = SplitExplicitFreeSurface(grid; substeps=10, extended_halos = false)

coriolis = FPlane(f=1e-4)

model = HydrostaticFreeSurfaceModel(; grid,
                                      coriolis,
                                      free_surface)

gaussian(x, L) = exp(-x^2 / 2L^2)

U = 0.1 # geostrophic velocity
L = Lh / 40 # gaussian width
x₀ = Lh / 4 # gaussian center
vᵍ(x, y, z) = -U * (x - x₀) / L * gaussian(x - x₀, L)

g = model.free_surface.gravitational_acceleration
η = model.free_surface.η
η₀ = coriolis.f * U * L / g # geostrophic free surface amplitude

ηᵍ(x) = η₀ * gaussian(x - x₀, L)
ηⁱ(x, y, z) = 2 * ηᵍ(x)

set!(model, v = vᵍ)
set!(model, η = ηⁱ)

gravity_wave_speed = sqrt(g * Lz) # hydrostatic (shallow water) gravity wave speed
Δt = 2 * getregion(model.grid, 1).Δxᶜᵃᵃ / gravity_wave_speed
simulation = Simulation(model; Δt, stop_iteration = 1000)

ut = []
vt = []
ηt = []

save_u(sim) = push!(ut, deepcopy(sim.model.velocities.u.data.regional_objects))  
save_v(sim) = push!(vt, deepcopy(sim.model.velocities.v.data.regional_objects))
save_η(sim) = push!(ηt, deepcopy(sim.model.free_surface.η.data.regional_objects))

function progress_message(sim) 
    @info @sprintf("[%.2f%%], iteration: %d, time: %.3f, max|w|: %.2e",
    100 * sim.model.clock.time / sim.stop_time, sim.model.clock.iteration,
    sim.model.clock.time, maximum(abs, sim.model.velocities.u))
end

simulation.callbacks[:save_η]   = Callback(save_η, IterationInterval(1))
simulation.callbacks[:save_v]   = Callback(save_v, IterationInterval(1))
simulation.callbacks[:save_u]   = Callback(save_u, IterationInterval(1))
simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(10))

run!(simulation)

# jldsave("variables_rank$(rank).jld2", v=vt, η=ηt, u=ut)

