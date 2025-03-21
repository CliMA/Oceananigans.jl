using Oceananigans
using Oceananigans.Units

arch = CPU()
Nx = Ny = Nz = 64
Lx = Ly = 512
Lz = 256
grid = RectilinearGrid(arch, size=(Nx, Ny, Nz), x=(0, Lx), y=(0, Ly), z=(0, Lz))

N²ₛ = 1e-4
ϵ = 0.8
Dᴴ = ϵ * N²ₛ * Lz
Mᴴ = Dᴴ - N²ₛ * Lz

buoyancy = Oceananigans.BuoyancyFormulations.MoistConditionalBuoyancy(N²ₛ)
model = NonhydrostaticModel(; grid, buoyancy, tracers=(:D, :M), advection=WENO())

Dᵢ(x, y, z) = Dᴴ * z + 1e-2 * Dᴴ * Lz * randn()
Mᵢ(x, y, z) = Mᴴ * z + 1e-2 * Mᴴ * Lz * randn()
set!(model, D=Dᵢ, M=Mᵢ)

simulation = Simulation(model, Δt=1, stop_time=4hours)
conjure_time_step_wizard!(simulation, cfl=0.2)

progress(sim) = @info string(iteration(sim),
                             ": time = ", prettytime(sim),
                             ", Δt = ", prettytime(sim.Δt),
                             ", max|w| = ", maximum(abs, model.velocities.w))

add_callback!(simulation, progress, IterationInterval(10))

ow = JLD2Writer(model, merge(model.velocities, model.tracers),
                schedule = TimeInterval(5minutes),
                filename = "moist_convection.jld2",
                indices = (:, 1, :), 
                overwrite_existing = true)

simulation.output_writers[:slice] = ow

run!(simulation)

using GLMakie

filename = "moist_convection.jld2"
wt = FieldTimeSeries(filename, "w")

fig = Figure()
axw = Axis(fig[1, 1])
slider = Slider(fig[2, 1], range=1:length(wt), startvalue=1)
n = slider.value
wn = @lift wt[$n]
heatmap!(axw, wn)

