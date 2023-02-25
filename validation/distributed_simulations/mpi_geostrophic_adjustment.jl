# Run this script with
#
# $ mpiexec -n 2 julia --project mpi_nonhydrostatic_two_dimensional_turbulence.jl
#
# for example.
#
# You also probably should set
#
# $ export JULIA_NUM_THREADS=1

using MPI
using Oceananigans
using Oceananigans.Distributed
using Oceananigans.Grids: topology, architecture
using Oceananigans.Units: kilometers, meters
using Printf
using JLD2

MPI.Init()

comm   = MPI.COMM_WORLD
rank   = MPI.Comm_rank(comm)
Nranks = MPI.Comm_size(comm)

topo = (Bounded, Periodic, Bounded)
arch = DistributedArch(CPU(); topology = topo, 
                 ranks=(Nranks, 1, 1),
                 use_buffers = true)

Lh = 100kilometers
Lz = 400meters

grid = RectilinearGrid(arch,
                       size = (80, 3, 1),
                       x = (0, Lh), y = (0, Lh), z = (-Lz, 0),
                       topology = topo)

bottom(x, y) = x > 80kilometers && x < 90kilometers ? 100.0 : -500meters
grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom))

coriolis = FPlane(f = 1e-4)

free_surface = SplitExplicitFreeSurface(; substeps = 10)

model = HydrostaticFreeSurfaceModel(; grid,
                                      coriolis = coriolis,
                                      free_surface = free_surface)

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

gravity_wave_speed = sqrt(g * grid.Lz) # hydrostatic (shallow water) gravity wave speed
wave_propagation_time_scale = model.grid.Δxᶜᵃᵃ / gravity_wave_speed
simulation = Simulation(model, Δt = 2wave_propagation_time_scale, stop_iteration = 1000)

outputs = Dict()

indices = (:, :, 1)

ηarr = Vector{Field}(undef, Int(simulation.stop_iteration) + 1)
varr = Vector{Field}(undef, Int(simulation.stop_iteration) + 1)
uarr = Vector{Field}(undef, Int(simulation.stop_iteration) + 1)

save_η(sim) = ηarr[sim.model.clock.iteration + 1] = deepcopy(sim.model.free_surface.η) 
save_v(sim) = varr[sim.model.clock.iteration + 1] = deepcopy(sim.model.velocities.v)   
save_u(sim) = uarr[sim.model.clock.iteration + 1] = deepcopy(sim.model.velocities.u)   

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

jldsave("variables_rank$(rank).jld2", varr = varr, ηarr = ηarr, uarr = uarr)

MPI.Finalize()
