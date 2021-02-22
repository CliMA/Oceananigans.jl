# # Geostrophic adjustment using Oceananigans.HydrostaticFreeSurfaceModel
#
# ## Install dependencies
#
# First let's make sure we have all required packages installed.

# ```julia
# using Pkg
# pkg"add Oceananigans, JLD2, Plots"
# ```

# ## A spherical domain
#
# We use a one-dimensional domain of geophysical proportions,

using Oceananigans
using Oceananigans.Grids: RegularLatitudeLongitudeGrid

#grid = RegularLatitudeLongitudeGrid(size = (720, 120, 1), latitude = (0, 60), longitude = (-180, 180), z = (-1, 0))
grid = RegularLatitudeLongitudeGrid(size = (360, 60, 1), latitude = (0, 60), longitude = (-180, 180), z = (-1, 0))

using Oceananigans.Coriolis: HydrostaticSphericalCoriolis

coriolis = HydrostaticSphericalCoriolis()

# ## Building a `HydrostaticFreeSurfaceModel`
#
# We use `grid` and `coriolis` to build a simple `HydrostaticFreeSurfaceModel`,

using Oceananigans.Models.HydrostaticFreeSurfaceModels: VectorInvariant
using Oceananigans.Models: HydrostaticFreeSurfaceModel
using Oceananigans.TurbulenceClosures: HorizontallyCurvilinearAnisotropicDiffusivity

closure = HorizontallyCurvilinearAnisotropicDiffusivity(νh=1, κh=1)

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    momentum_advection = VectorInvariant(),
                                    tracers = (),
                                    buoyancy = nothing,
                                    coriolis = coriolis,
                                    closure = closure)

# ## A geostrophic adjustment initial value problem
#
# We pose a geostrophic adjustment problem that consists of a partially-geostrophic
# Gaussian height field complemented by a geostrophic ``y``-velocity,

Gaussian(λ, ϕ, dλ, dϕ) = exp(-λ^2 / 2dλ^2 - ϕ^2 / 2dϕ^2)

a = 0.01 # geostrophic velocity
L = 4 # degree
ϕ₀ = 30 # degrees

splash(λ, ϕ, z) = a * Gaussian(λ, ϕ - ϕ₀, L, L)

set!(model, η=splash)

# ## Running a `Simulation`
#
# We pick a time-step that resolves the surface dynamics,

g = model.free_surface.gravitational_acceleration

gravity_wave_speed = sqrt(g * grid.Lz) # hydrostatic (shallow water) gravity wave speed

wave_propagation_time_scale = 100 # 100e3 * model.grid.Δλ / gravity_wave_speed

simulation = Simulation(model, Δt = 0.1 * wave_propagation_time_scale, stop_iteration = 1000)

# ## Output
#
# We output the velocity field and free surface displacement,

output_fields = merge(model.velocities, (η=model.free_surface.η,))

using Oceananigans.OutputWriters: JLD2OutputWriter, IterationInterval

simulation.output_writers[:fields] = JLD2OutputWriter(model, output_fields,
                                                      schedule = IterationInterval(10),
                                                      prefix = "rossby_splash",
                                                      force = true)

run!(simulation)

# ## Visualizing the results

using JLD2, Printf, Oceananigans.Grids, Plots

λ, ϕ, r = nodes(model.free_surface.η, reshape=true)

λ = λ .+ 180
ϕ = ϕ .+ 90

using Oceananigans.Utils: hours

file = jldopen(simulation.output_writers[:fields].filepath)

iterations = parse.(Int, keys(file["timeseries/t"]))

iter = iterations[end]
η = file["timeseries/η/$iter"][:, :, 1]

x = @. grid.radius * cosd(ϕ) * sind(λ)
y = @. grid.radius * sind(ϕ) * sind(λ)
z = @. grid.radius * cosd(λ) * ϕ ./ ϕ

x = x[:, :, 1]'
y = y[:, :, 1]'
z = z[:, :, 1]'

Makie.surface(x, y, z, color=η)
