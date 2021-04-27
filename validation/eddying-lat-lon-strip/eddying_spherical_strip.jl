# # Barotropic gyre

using Oceananigans
using Oceananigans.Grids

using Oceananigans.Coriolis:
    HydrostaticSphericalCoriolis,
    VectorInvariantEnergyConserving,
    VectorInvariantEnstrophyConserving

using Oceananigans.Models.HydrostaticFreeSurfaceModels:
    HydrostaticFreeSurfaceModel,
    VerticalVorticityField,
    VectorInvariant,
    ExplicitFreeSurface,
    ImplicitFreeSurface

using Oceananigans.TurbulenceClosures: HorizontallyCurvilinearAnisotropicDiffusivity
using Oceananigans.Utils: prettytime, hours, day, days, years
using Oceananigans.OutputWriters: JLD2OutputWriter, TimeInterval, IterationInterval

using Statistics
using JLD2
using Printf

Nx = 60
Ny = 60

# A spherical domain
grid = RegularLatitudeLongitudeGrid(size = (Nx, Ny, 1),
                                    longitude = (-180, 180),
                                    latitude = (-80, 80),
                                    z = (-4000, 0))

#free_surface = ImplicitFreeSurface(gravitational_acceleration=0.1)
free_surface = ExplicitFreeSurface(gravitational_acceleration=0.1)

coriolis = HydrostaticSphericalCoriolis(scheme = VectorInvariantEnstrophyConserving())

@show const νh₀ = 5e3 * (60 / grid.Nx)^2
@inline νh(λ, φ, z, t) = νh₀ * cos(π * φ / 180)
variable_horizontal_diffusivity = HorizontallyCurvilinearAnisotropicDiffusivity(νh=νh)

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    architecture = CPU(),
                                    momentum_advection = VectorInvariant(),
                                    free_surface = free_surface,
                                    coriolis = coriolis,
                                    closure = variable_horizontal_diffusivity,
                                    tracers = nothing,
                                    buoyancy = nothing)

g = model.free_surface.gravitational_acceleration

gravity_wave_speed = sqrt(g * grid.Lz) # hydrostatic (shallow water) gravity wave speed

# Time-scale for gravity wave propagation across the smallest grid cell
wave_propagation_time_scale = min(grid.radius * cosd(maximum(abs, grid.φᵃᶜᵃ)) * deg2rad(grid.Δλ),
                                  grid.radius * deg2rad(grid.Δφ)) / gravity_wave_speed

set!(model,
     u = (x, y, z) -> rand(),
     v = (x, y, z) -> rand())

# Zero out mean motion
model.velocities.u .-= mean(model.velocities.u)
model.velocities.v .-= mean(model.velocities.v)

# Set target velocity to fraction of free surface velocity
max_u = maximum(model.velocities.u)
max_v = maximum(model.velocities.v)
max_speed = sqrt(max_u^2 + max_v^2)

target_speed = 0.1 * gravity_wave_speed
model.velocities.u ./= target_speed / max_speed
model.velocities.v ./= target_speed / max_speed

mutable struct Progress
    interval_start_time :: Float64
end

function (p::Progress)(sim)
    wall_time = (time_ns() - p.interval_start_time) * 1e-9

    @info @sprintf("Time: %s, iteration: %d, max(u): %.2e m s⁻¹, wall time: %s",
                   prettytime(sim.model.clock.time),
                   sim.model.clock.iteration,
                   maximum(sim.model.velocities.u),
                   prettytime(wall_time))

    p.interval_start_time = time_ns()

    return nothing
end


ζ = VerticalVorticityField(model)
compute!(ζ)

Δt = 0.1wave_propagation_time_scale

@info """
    Maximum vertical vorticity: $(maximum(ζ))
    Inverse maximum vertical vorticity: $(prettytime(1/maximum(ζ)))
    Minimum wave propagation time scale: $(prettytime(wave_propagation_time_scale))
    Time step: $(prettytime(Δt))
"""

simulation = Simulation(model,
                        Δt = Δt,
                        stop_time = 1years,
                        iteration_interval = 100,
                        progress = Progress(time_ns()))

output_fields = merge(model.velocities, (η=model.free_surface.η,))

output_prefix = "barotropic_gyre_Nx$(grid.Nx)_Ny$(grid.Ny)"

simulation.output_writers[:fields] = JLD2OutputWriter(model, output_fields,
                                                      schedule = TimeInterval(10day),
                                                      prefix = output_prefix,
                                                      field_slicer = nothing,
                                                      force = true)

run!(simulation)

#####
##### Animation!
#####

include("visualize.jl")

visualize_plots(simulation.output_writers[:fields].filepath)
