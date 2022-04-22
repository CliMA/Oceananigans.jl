using Statistics
using Logging
using Printf
using DataDeps
using JLD2

using Oceananigans
using Oceananigans.Units
using Oceananigans.Coriolis
using Oceananigans.Models.HydrostaticFreeSurfaceModels
using Oceananigans.TurbulenceClosures

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

Logging.global_logger(OceananigansLogger())

dd = DataDep("cubed_sphere_32_grid",
    "Conformal cubed sphere grid with 32×32 grid points on each face",
    "https://github.com/CliMA/OceananigansArtifacts.jl/raw/main/cubed_sphere_grids/cubed_sphere_32_grid.jld2",
    "b1dafe4f9142c59a2166458a2def743cd45b20a4ed3a1ae84ad3a530e1eff538" # sha256sum
)

DataDeps.register(dd)

## Choose a lat-lon grid or cubed sphere face grid

H = 4kilometers

# grid = LatitudeLongitudeGrid(size = (60, 60, 1), longitude = (-40, 40), latitude = (-40, 40), z = (-H, 0))

cs32_filepath = datadep"cubed_sphere_32_grid/cubed_sphere_32_grid.jld2"
grid = ConformalCubedSphereFaceGrid(cs32_filepath, face=1, Nz=1, z=(-H, 0))

## Turbulent diffusivity closure

const νh₀ = 5e3 * (60 / grid.Nx)^2

@inline νh(λ, φ, z, t) = νh₀ * cos(π * φ / 180)

variable_horizontal_diffusivity = HorizontalScalarDiffusivity(ν=νh)
constant_horizontal_diffusivity = HorizontalScalarDiffusivity(ν=νh₀)

## Model setup

model = HydrostaticFreeSurfaceModel(
          architecture = CPU(),
                  grid = grid,
    momentum_advection = VectorInvariant(),
          free_surface = ExplicitFreeSurface(gravitational_acceleration=0.1),
        # free_surface = ImplicitFreeSurface(gravitational_acceleration=0.1)
           #  coriolis = nothing,
              coriolis = HydrostaticSphericalCoriolis(scheme = VectorInvariantEnstrophyConserving()),
               closure = nothing,
             # closure = constant_horizontal_diffusivity,
             # closure = variable_horizontal_diffusivity,
               tracers = nothing,
              buoyancy = nothing
)

## Very small sea surface height perturbation so the resulting dynamics are well-described
## by a linear free surface.

A  = 1e-5 * H  # Amplitude of the perturbation
λ₀ = 0   # Central longitude
φ₀ = 20  # Central latitude
Δλ = 10  # Longitudinal width
Δφ = 10  # Latitudinal width

η′(λ, φ, z) = A * exp(- (λ - λ₀)^2 / Δλ^2) * exp(- (φ - φ₀)^2 / Δφ^2)

set!(model, η=η′)

# g = model.free_surface.gravitational_acceleration
# gravity_wave_speed = sqrt(g * H) # hydrostatic (shallow water) gravity wave speed

# # Time-scale for gravity wave propagation across the smallest grid cell
# wave_propagation_time_scale = min(grid.radius * cosd(maximum(abs, grid.ϕᵃᶜᵃ)) * deg2rad(grid.Δλ),
#                                   grid.radius * deg2rad(grid.Δϕ)) / gravity_wave_speed

mutable struct Progress
    interval_start_time :: Float64
end

function (p::Progress)(sim)
    wall_time = (time_ns() - p.interval_start_time) * 1e-9

    @info @sprintf("Time: %s, iteration: %d, max(u⃗): (%.2e, %.2e) m/s, extrema(η): (min=%.2e, max=%.2e), wall time: %s",
                   prettytime(sim.model.clock.time),
                   sim.model.clock.iteration,
                   maximum(abs, sim.model.velocities.u),
                   maximum(abs, sim.model.velocities.v),
                   minimum(abs, sim.model.free_surface.η),
                   maximum(abs, sim.model.free_surface.η),
                   prettytime(wall_time))

    p.interval_start_time = time_ns()

    return nothing
end

simulation = Simulation(model,
                        Δt = 20minutes,
                        stop_time = 30days,
                        iteration_interval = 20,
                        progress = Progress(time_ns()))

output_fields = merge(model.velocities, (η=model.free_surface.η,))

output_prefix = grid isa LatitudeLongitudeGrid ? "lat_lon_waves" : "cubed_sphere_face_waves"

simulation.output_writers[:fields] = JLD2OutputWriter(model, output_fields,
                                                      schedule = TimeInterval(1hour),
                                                      prefix = output_prefix,
                                                      overwrite_existing = true)

run!(simulation)

# #####
# ##### Animation!
# #####

# include("visualize_barotropic_gyre.jl")

# visualize_barotropic_gyre(simulation.output_writers[:fields])
