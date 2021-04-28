# # Freely decaying barotropic turbulence on a latitude-longitude strip

using Oceananigans
using Oceananigans.Grids

using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: FunctionField

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

using Oceananigans.TurbulenceClosures:
    HorizontallyCurvilinearAnisotropicDiffusivity,
    HorizontallyCurvilinearAnisotropicBiharmonicDiffusivity

using Oceananigans.Utils: prettytime, hours, day, days, years, year
using Oceananigans.OutputWriters: JLD2OutputWriter, TimeInterval, IterationInterval

using Statistics
using JLD2
using Printf

using Oceananigans.AbstractOperations: AbstractGridMetric, _unary_operation

using CUDA
CUDA.math_mode!(CUDA.FAST_MATH)

#####
##### Grid
#####

latitude = (-80, 80)
Δφ = latitude[2] - latitude[1]

resolution = 1/2 # degree
Nx = round(Int, 360 / resolution)
Ny = round(Int, Δφ / resolution)

# A spherical domain
@show grid = RegularLatitudeLongitudeGrid(size = (Nx, Ny, 1),
                                          longitude = (-180, 180),
                                          latitude = latitude,
                                          halo = (2, 2, 2),
                                          z = (-100, 0))

#####
##### Physics and model setup
#####

free_surface = ExplicitFreeSurface(gravitational_acceleration=1.0)

equator_Δx = grid.radius * deg2rad(grid.Δλ)
diffusive_time_scale = 60days

@show const νh₂₀ =        equator_Δx^2 / diffusive_time_scale
@show const νh₄₀ = 1e-6 * equator_Δx^4 / diffusive_time_scale

@inline νh₂(λ, φ, z, t) = νh₂₀ * cos(π * φ / 180)
@inline νh₄(λ, φ, z, t) = νh₄₀ * cos(π * φ / 180)

variable_horizontal_diffusivity = HorizontallyCurvilinearAnisotropicDiffusivity(νh=νh₂)
variable_horizontal_biharmonic_diffusivity = HorizontallyCurvilinearAnisotropicBiharmonicDiffusivity(νh=νh₄)

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    architecture = GPU(),
                                    momentum_advection = VectorInvariant(),
                                    free_surface = free_surface,
                                    coriolis = nothing,
                                    tracers = nothing,
                                    buoyancy = nothing,
                                    #closure = variable_horizontal_diffusivity)
                                    closure = variable_horizontal_biharmonic_diffusivity)

#####
##### Initial condition
#####

ψ = Field(Face, Face, Center, model.architecture, model.grid)
set!(ψ, (x, y, z) -> rand())
fill_halo_regions!(ψ, model.architecture)

u, v, w = model.velocities

u .= - ∂y(ψ)
v .= + ∂x(ψ)


#####
##### Rescale velocity to fraction of free surface velocity
#####

# Time-scale for gravity wave propagation across the smallest grid cell
g = model.free_surface.gravitational_acceleration
gravity_wave_speed = sqrt(g * grid.Lz) # hydrostatic (shallow water) gravity wave speed

minimum_Δx = grid.radius * cosd(maximum(abs, grid.φᵃᶜᵃ)) * deg2rad(grid.Δλ)
minimum_Δy = grid.radius * deg2rad(grid.Δφ)
wave_propagation_time_scale = min(minimum_Δx, minimum_Δy) / gravity_wave_speed

@show max_u = maximum(u)
@show max_v = maximum(v)
max_speed_ish = sqrt(max_u^2 + max_v^2)

@show target_speed = 0.5 * gravity_wave_speed
u .*= target_speed / max_speed_ish
v .*= target_speed / max_speed_ish

@show maximum(u)
@show maximum(v)

# Zero out mean motion
using Oceananigans.AbstractOperations: volume
using Oceananigans.Fields: ReducedField

u_dV = u * volume
mean_u = ReducedField(nothing, nothing, nothing, model.architecture, model.grid, dims=(1, 2, 3))
mean!(mean_u, u_dV)

v_dV = v * volume
mean_v = ReducedField(nothing, nothing, nothing, model.architecture, model.grid, dims=(1, 2, 3))
mean!(mean_v, v_dV)

model.velocities.u .-= mean_u
model.velocities.v .-= mean_v

#####
##### Simulation setup
#####

ζ = VerticalVorticityField(model)
compute!(ζ)
Δt = 0.1 * minimum_Δx / target_speed

@info """
    Maximum vertical vorticity: $(maximum(ζ))
    Inverse maximum vertical vorticity: $(prettytime(1/maximum(ζ)))
    Minimum wave propagation time scale: $(prettytime(wave_propagation_time_scale))
    Time step: $(prettytime(Δt))
"""

mutable struct Progress; interval_start_time::Float64; end

function (p::Progress)(sim)
    wall_time = (time_ns() - p.interval_start_time) * 1e-9

    compute!(ζ)

    @info @sprintf("Time: %s, iteration: %d, max(u): %.2e m s⁻¹, wall time: %s",
                   prettytime(sim.model.clock.time),
                   sim.model.clock.iteration,
                   maximum(abs, ζ),
                   prettytime(wall_time))

    p.interval_start_time = time_ns()

    return nothing
end


simulation = Simulation(model,
                        Δt = Δt,
                        stop_time = 2year,
                        iteration_interval = 1000,
                        progress = Progress(time_ns()))

output_fields = merge(model.velocities, (η=model.free_surface.η, ζ=ζ))

output_prefix = "freely_decaying_barotropic_turbulence_Nx$(grid.Nx)_Ny$(grid.Ny)"

simulation.output_writers[:fields] = JLD2OutputWriter(model, (ζ = ζ,),
                                                      schedule = TimeInterval(30day),
                                                      prefix = output_prefix,
                                                      force = true)

# Let's goo!
run!(simulation)
