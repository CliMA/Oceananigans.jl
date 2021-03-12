# # Barotropic gyre

using Oceananigans
using Oceananigans.Grids

using Oceananigans.Coriolis:
    HydrostaticSphericalCoriolis,
    VectorInvariantEnergyConserving,
    VectorInvariantEnstrophyConserving

using Oceananigans.Models.HydrostaticFreeSurfaceModels:
    HydrostaticFreeSurfaceModel,
    VectorInvariant,
    ExplicitFreeSurface

using Oceananigans.TurbulenceClosures: HorizontallyCurvilinearAnisotropicDiffusivity
using Oceananigans.Utils: prettytime, hours, day, days, years
using Oceananigans.OutputWriters: JLD2OutputWriter, TimeInterval, IterationInterval

using Statistics
using JLD2
using Printf

function geographic2cartesian(λ, φ, radius=1)
    Nx = length(λ)
    Ny = length(φ)

    λ = repeat(reshape(λ, Nx, 1), 1, Ny) 
    φ = repeat(reshape(φ, 1, Ny), Nx, 1)

    λ_azimuthal = λ .+ 180  # Convert to λ ∈ [0°, 360°]
    φ_azimuthal = 90 .- φ   # Convert to φ ∈ [0°, 180°] (0° at north pole)

    x = @. radius * cosd(λ_azimuthal) * sind(φ_azimuthal)
    y = @. radius * sind(λ_azimuthal) * sind(φ_azimuthal)
    z = @. radius * cosd(φ_azimuthal)

    return x, y, z
end

#=

Nx = 1 * 60
Ny = 1 * 60

# A spherical domain
grid = RegularLatitudeLongitudeGrid(size = (Nx, Ny, 1),
                                    longitude = (-30, 30),
                                    latitude = (15, 75),
                                    z = (-4000, 0))

free_surface = ExplicitFreeSurface(gravitational_acceleration=0.1)

coriolis = HydrostaticSphericalCoriolis(scheme = VectorInvariantEnstrophyConserving())

@show surface_wind_stress_parameters = (τ₀ = 1e-4,
                                        Lφ = grid.Ly,
                                        φ₀ = 15)

surface_wind_stress(λ, φ, t, p) = p.τ₀ * cos(2π * (φ - p.φ₀) / p.Lφ)

surface_wind_stress_bc = BoundaryCondition(Flux,
                                           surface_wind_stress,
                                           parameters = surface_wind_stress_parameters)

μ = 1 / 60days

@inline u_bottom_drag(i, j, grid, clock, fields, μ) = @inbounds - μ * fields.u[i, j, 1]
@inline v_bottom_drag(i, j, grid, clock, fields, μ) = @inbounds - μ * fields.v[i, j, 1]

u_bottom_drag_bc = BoundaryCondition(Flux,
                                     u_bottom_drag,
                                     discrete_form = true,
                                     parameters = μ)

v_bottom_drag_bc = BoundaryCondition(Flux,
                                     v_bottom_drag,
                                     discrete_form = true,
                                     parameters = μ)

u_bcs = UVelocityBoundaryConditions(grid,
                                    top = surface_wind_stress_bc,
                                    bottom = u_bottom_drag_bc)

v_bcs = VVelocityBoundaryConditions(grid,
                                    bottom = v_bottom_drag_bc)
                                        
@show const νh₀ = 5e3 * (60 / grid.Nx)^2

@inline νh(λ, φ, z, t) = νh₀ * cos(π * φ / 180)

variable_horizontal_diffusivity = HorizontallyCurvilinearAnisotropicDiffusivity(νh=νh)

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    architecture = CPU(),
                                    momentum_advection = VectorInvariant(),
                                    free_surface = free_surface,
                                    coriolis = coriolis,
                                    boundary_conditions = (u=u_bcs, v=v_bcs),
                                    closure = variable_horizontal_diffusivity,
                                    buoyancy = nothing)

g = model.free_surface.gravitational_acceleration

gravity_wave_speed = sqrt(g * grid.Lz) # hydrostatic (shallow water) gravity wave speed

# Time-scale for gravity wave propagation across the smallest grid cell
wave_propagation_time_scale = min(grid.radius * cosd(maximum(abs, grid.ϕᵃᶜᵃ)) * deg2rad(grid.Δλ),
                                  grid.radius * deg2rad(grid.Δϕ)) / gravity_wave_speed

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

simulation = Simulation(model,
                        Δt = 0.2wave_propagation_time_scale,
                        stop_time = 3years,
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
=#

#####
##### Animation!
#####

using GLMakie

include("visualize_barotropic_gyre.jl")

visualize_barotropic_gyre(simulation.output_writers[:fields])
