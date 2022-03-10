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
using Oceananigans.Grids: min_Δy, min_Δx
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

# In addition to the solid body rotation solution, we paint a Gaussian tracer patch
# on the spherical strip to visualize the rotation.
const U = 0.1

northern_boundary = 80 # degrees
Ω = 1 # rad / s
g = 1 # m s⁻²

Nx=512; Ny=512; dev=(0, 1); architecture = GPU()

function run_solid_body_tracer_advection(; architecture = CPU(),
                                           Nx = 360,
                                           Ny = 8,
                                           dev = nothing)

    # A spherical domain
    grid = RectilinearGrid(architecture, size = (Nx, Ny),
                                       halo = (3, 3),
                                       topology = (Periodic, Periodic, Flat),
                                       x = (0, 1),
                                       y = (0, 1))

    if dev isa Nothing
        mrg = grid
    else
        mrg = MultiRegionGrid(grid, partition = XPartition(2), devices = dev)
    end

    uᵢ = Field{Face, Center, Center}(grid)
    fill!(uᵢ, U)

    model = HydrostaticFreeSurfaceModel(grid = mrg,
                                        tracers = (:c, :d, :e),
                                        velocities = PrescribedVelocityFields(u=uᵢ),
                                        free_surface = ExplicitFreeSurface(),
                                        momentum_advection = nothing,
                                        tracer_advection = WENO5(),
                                        coriolis = nothing,
                                        buoyancy = nothing,
                                        closure  = nothing)

    # Tracer patch for visualization
    Gaussian(x, y, L) = exp(-(x^2 + y^2) / 2L^2)

    # Tracer patch parameters
    L = 0.1 # degree

    cᵢ(x, y, z) = Gaussian(x, 0, L)
    dᵢ(x, y, z) = Gaussian(0, y, L)
    eᵢ(x, y, z) = Gaussian(x, y, L)

    set!(model, c=cᵢ, d=dᵢ, e=eᵢ)

    Δx_min = min_Δx(grid)
    Δy_min = min_Δy(grid)
    Δ_min = min(Δx_min, Δy_min)

    # Time-scale for tracer advection across the smallest grid cell
    @show advection_time_scale = Δ_min / U
    super_rotation_period = 200advection_time_scale

    Δt = 0.1advection_time_scale
    simulation = Simulation(model,
                            Δt = Δt,
                            stop_time = super_rotation_period)

                            # stop_time = super_rotations * super_rotation_period)

    progress(sim) = @info(@sprintf("Iter: %d, time: %.1f, Δt: %.3f, max|c|: %.2f",
                                   sim.model.clock.iteration, sim.model.clock.time,
                                   sim.Δt, maximum(abs, sim.model.tracers.c)))

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

    run!(simulation)

    @show simulation.run_wall_time
    return simulation
end

simulation_serial   = run_solid_body_tracer_advection(architecture=GPU(), Nx=256, Ny=256)
simulation_parallel = run_solid_body_tracer_advection(Nx=256, Ny=256, dev=(0, 1))

# model2 = run_solid_body_tracer_advection(Nx=256, Ny=64, multigpu=true, dev = (2, 3), super_rotations=0.01)
# model0 = run_solid_body_tracer_advection(Nx=128, Ny=64, super_rotations=0.01, architecture=GPU())
# model1 = run_solid_body_tracer_advection(Nx=128, Ny=64, multigpu=true, dev = (3, ), super_rotations=0.01, architecture=GPU())
