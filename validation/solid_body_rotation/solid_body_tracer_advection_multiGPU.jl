# # Solid body rotation of a meridional sector on the sphere
#
# This script implements the "Global Steady State Nonlinear Zonal Geostrophic Flow"
# validation experiment from
#
# > Williamson et al., "A Standard Test Set for Numerical Approximations to the Shallow
#   Water Equations in Spherical Geometry", Journal of Computational Physics, 1992.
#
# The problem is posed in spherical strip between 60ᵒS and 60ᵒN latitude on a sphere with
# unit radius.
#
# # Dependencies
#
# The validation experiment depends on Oceananigans, JLD2, Printf, and GLMakie for visualization

using Oceananigans
using Oceananigans.Grids
using Oceananigans.MultiRegion

using Oceananigans.Coriolis:
    HydrostaticSphericalCoriolis,
    VectorInvariantEnergyConserving,
    VectorInvariantEnstrophyConserving

using Oceananigans.Models.HydrostaticFreeSurfaceModels:
    HydrostaticFreeSurfaceModel,
    VectorInvariant,
    ExplicitFreeSurface,
    PrescribedVelocityFields

using Oceananigans.TurbulenceClosures

using Oceananigans.Utils: prettytime, hours
using Oceananigans.OutputWriters: JLD2OutputWriter, TimeInterval, IterationInterval

using JLD2
using Printf

using BenchmarkTools
# using GLMakie

# # The geostrophic flow
#
# ```math
# u = U \cos ϕ
# v = 0
# η = - g^{-1} \left (R Ω U + \frac{U^2}{2} \right ) \sin^2 ϕ
# ```
#
# is a steady nonlinear flow on a sphere of radius ``R`` with gravitational
# acceleration ``g``, corresponding to solid body rotation
# in the same direction as the "background" rotation rate ``\Omega``.
# The velocity ``U`` determines the magnitude of the additional rotation.

const U = 0.1

solid_body_rotation(λ, ϕ) = U * cosd(ϕ)
solid_body_geostrophic_height(λ, ϕ, R, Ω, g) = (R * Ω * U + U^2 / 2) * sind(ϕ)^2 / g

# In addition to the solid body rotation solution, we paint a Gaussian tracer patch
# on the spherical strip to visualize the rotation.

northern_boundary = 80 # degrees
Ω = 1 # rad / s
g = 1 # m s⁻²

function run_solid_body_tracer_advection(; architecture = CPU(),
                                           multigpu = false,
                                           Nx = 360,
                                           Ny = 8,
                                           dev = nothing,
                                           super_rotations = 4)

    # A spherical domain
    @show grid = LatitudeLongitudeGrid(architecture, size = (Nx, Ny, 1),
                                       radius = 1,
                                       halo = (3, 3, 3),
                                       latitude = (-northern_boundary, northern_boundary),
                                       longitude = (-180, 180),
                                       z = (-1, 0))

    if multigpu
        mrg = MultiRegionGrid(grid, partition = XPartition(2), devices = dev)
    else
        mrg = grid
    end

    uᵢ(λ, ϕ, z, t=0) = solid_body_rotation(λ, ϕ)

    model = HydrostaticFreeSurfaceModel(grid = mrg,
                                        tracers = (:c, :d, :e),
                                        # velocities = PrescribedVelocityFields(u=uᵢ),
                                        momentum_advection = VectorInvariant(),
                                        tracer_advection = WENO5(),
                                        coriolis = nothing,
                                        buoyancy = nothing,
                                        closure  = nothing)
    return model
end
#     # Tracer patch for visualization
#     Gaussian(λ, ϕ, L) = exp(-(λ^2 + ϕ^2) / 2L^2)

#     # Tracer patch parameters
#     L = 24 # degree
#     ϕ₀ = 0 # degrees

#     cᵢ(λ, ϕ, z) = Gaussian(λ, 0, L)
#     dᵢ(λ, ϕ, z) = Gaussian(0, ϕ - ϕ₀, L)
#     eᵢ(λ, ϕ, z) = Gaussian(λ, ϕ - ϕ₀, L)

#     set!(model, c=cᵢ, d=dᵢ, e=eᵢ)

#     ϕᵃᶜᵃ_max = maximum(abs, ynodes(Center, grid))
#     Δx_min = grid.radius * cosd(ϕᵃᶜᵃ_max) * deg2rad(grid.Δλᶜᵃᵃ)
#     Δy_min = grid.radius * deg2rad(grid.Δφᵃᶜᵃ)
#     Δ_min = min(Δx_min, Δy_min)

#     # Time-scale for tracer advection across the smallest grid cell
#     @show advection_time_scale = Δ_min / U
#     super_rotation_period = 2π * grid.radius / U

#     Δt = 0.1advection_time_scale
#     simulation = Simulation(model,
#                             Δt = Δt,
#                             stop_time = 1000Δt)

#                             # stop_time = super_rotations * super_rotation_period)

#     progress(sim) = @info(@sprintf("Iter: %d, time: %.1f, Δt: %.3f, max|c|: %.2f",
#                                    sim.model.clock.iteration, sim.model.clock.time,
#                                    sim.Δt, maximum(abs, sim.model.tracers.c)))

#     simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

#     run!(simulation)

#     @show simulation.run_wall_time
#     return simulation
# end

# simulation_serial   = run_solid_body_tracer_advection(architecture=GPU(), Nx=512, Ny=256, super_rotations=0.01)
# simulation_parallel = run_solid_body_tracer_advection(Nx=512, Ny=512, multigpu=true, super_rotations=0.01)

# model2 = run_solid_body_tracer_advection(Nx=256, Ny=64, multigpu=true, dev = (2, 3), super_rotations=0.01)
# model0 = run_solid_body_tracer_advection(Nx=128, Ny=64, super_rotations=0.01, architecture=GPU())
# model1 = run_solid_body_tracer_advection(Nx=128, Ny=64, multigpu=true, dev = (3, ), super_rotations=0.01, architecture=GPU())
