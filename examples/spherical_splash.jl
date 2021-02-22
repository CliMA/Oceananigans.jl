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

grid = RegularLatitudeLongitudeGrid(size = (60, 360, 1), latitude = (0, 60), longitude = (-180, 180), z = (-1, 0))

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

xu, yu, zu = nodes(model.velocities.u)
xv, yv, zv = nodes(model.velocities.v)
xη, yη, zη = nodes(model.free_surface.η)

using Oceananigans.Utils: hours

file = jldopen(simulation.output_writers[:fields].filepath)

iterations = parse.(Int, keys(file["timeseries/t"]))

anim = @animate for (i, iter) in enumerate(iterations)

    u = file["timeseries/u/$iter"][:, :, 1]
    v = file["timeseries/v/$iter"][:, :, 1]
    η = file["timeseries/η/$iter"][:, :, 1]
    t = file["timeseries/t/$iter"]

    titlestr = @sprintf("Rossby wave...")

    #v_plot = Plots.contourf(xv, yv, v)
    #u_plot = Plots.contourf(xu, yu, u)
    #η_plot = Plots.contourf(xη, yη, η)

    u_plot = Plots.heatmap(u)
    v_plot = Plots.heatmap(v)
    η_plot = Plots.heatmap(η)

    Plots.plot(v_plot, u_plot, η_plot, layout = (1, 3), size = (800, 600))
end

close(file)

Plots.mp4(anim, "rossby_splash.mp4", fps = 15) # hide
