# # Solid body rotation of a meridional sector on the sphere
#
# This script implements the "Global Steady State Nonlinear Zonal Geostrophic Flow"
# validation experiment from
#
# > Williamson et al., "A Standard Test Set for Numerical Approximations to the Shallow
#   Water Equations in Spherical Geometry", Journal of Computational Physics, 1992.
#
# The problem is posed in spherical strip between 80ᵒS and 80ᵒN latitude on a sphere with
# unit radius.
#
# # Dependencies
#
# The validation experiment depends on Oceananigans, JLD2, Printf, and GLMakie for visualization

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

using Oceananigans.Utils: prettytime, hours
using Oceananigans.OutputWriters: JLD2OutputWriter, TimeInterval, IterationInterval

using Statistics
using JLD2
using Printf
using GLMakie 

# # The geostrophic flow
#
# ```math
# u = U \cos φ
# v = 0
# η = - g^{-1} \left (R Ω U + \frac{U^2}{2} \right ) \sin^2 φ
# ```
#
# is a steady nonlinear flow on a sphere of radius ``R`` with gravitational
# acceleration ``g``, corresponding to solid body rotation
# in the same direction as the "background" rotation rate ``\Omega``.
# The velocity ``U`` determines the magnitude of the additional rotation.

const U = 0.1

solid_body_rotation(φ) = U * cosd(φ)
solid_body_geostrophic_height(φ, R, Ω, g) = (R * Ω * U + U^2 / 2) * sind(φ)^2 / g

# In addition to the solid body rotation solution, we paint a Gaussian tracer patch
# on the spherical strip to visualize the rotation.

function run_solid_body_rotation(; architecture = CPU(),
                                   Nx = 90,
                                   Ny = 30,
                                   coriolis_scheme = VectorInvariantEnstrophyConserving(),
                                   advection_scheme = VectorInvariant(),
                                   super_rotations = 4,
                                   prefix = "vector_invariant")

    # A spherical domain
    grid = LatitudeLongitudeGrid(architecture, size = (Nx, Ny, 1),
                                 radius = 1,
                                 latitude = (-80, 80),
                                 halo = (3, 3, 3),
                                 longitude = (-180, 180),
                                 z = (-1, 0))

    free_surface = ExplicitFreeSurface(gravitational_acceleration = 1)

    coriolis = HydrostaticSphericalCoriolis(rotation_rate = 1,
                                            scheme = coriolis_scheme)

    model = HydrostaticFreeSurfaceModel(grid = grid,
                                        momentum_advection = advection_scheme,
                                        free_surface = free_surface,
                                        coriolis = coriolis,
                                        tracers = :c,
                                        buoyancy = nothing,
                                        closure = nothing)

    g = model.free_surface.gravitational_acceleration
    R = model.grid.radius
    Ω = model.coriolis.rotation_rate

    uᵢ(λ, φ, z) = solid_body_rotation(φ)
    ηᵢ(λ, φ)    = solid_body_geostrophic_height(φ, R, Ω, g)

    # Tracer patch for visualization
    Gaussian(λ, φ, L) = exp(-(λ^2 + φ^2) / 2L^2)

    # Tracer patch parameters
    L = 10 # degree
    φ₀ = 5 # degrees

    cᵢ(λ, φ, z) = Gaussian(λ, φ - φ₀, L)

    set!(model, u=uᵢ, η=ηᵢ, c=cᵢ)

    gravity_wave_speed = sqrt(g * grid.Lz) # hydrostatic (shallow water) gravity wave speed

    # Time-scale for gravity wave propagation across the smallest grid cell
    wave_propagation_time_scale = min(grid.radius * cosd(maximum(abs, grid.φᵃᶜᵃ)) * deg2rad(grid.Δλᶜᵃᵃ),
                                      grid.radius * deg2rad(grid.Δφᵃᶜᵃ)) / gravity_wave_speed

    super_rotation_period = 2π * grid.radius / U

    simulation = Simulation(model,
                            Δt = 0.5wave_propagation_time_scale,
                            stop_time = super_rotations * super_rotation_period)

    progress(sim) = @printf("Iter: %d, time: %s, Δt: %s, max|u|: %.3f, max|η|: %.3f \n",
                            iteration(sim), prettytime(sim), prettytime(sim.Δt), maximum(abs, model.velocities.u), maximum(abs, model.free_surface.η))

    simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

    output_fields = merge(model.velocities, model.tracers, (η=model.free_surface.η,))

    output_prefix = "solid_body_rotation_Nx$(grid.Nx)_" * prefix

    simulation.output_writers[:fields] = JLD2OutputWriter(model, output_fields,
                                                          schedule = TimeInterval(super_rotation_period / 1000),
                                                          prefix = output_prefix,
                                                          field_slicer = nothing,
                                                          force = true)

    run!(simulation)

    return simulation.output_writers[:fields].filepath
end

function plot_zonal_average_solid_body_rotation(filepath)
    @show output_prefix = basename(filepath)[1:end-5]

    file = jldopen(filepath)

    iterations = parse.(Int, keys(file["timeseries/t"]))

    Nx = file["grid/Nx"]
    Ny = file["grid/Ny"]

    grid = LatitudeLongitudeGrid(size = (Nx, Ny, 1),
                                 radius = 1,
                                 latitude = (-80, 80),
                                 longitude = (-180, 180),
                                 halo = (3, 3, 3),
                                 z = (-1, 0))

    super_rotation_period = 2π * grid.radius / U

    φ = ynodes(Center, grid)

    iter = Observable(0)

    plot_title = @lift @sprintf("Zonally-averaged velocity in solid body rotation: rotations = %.3f",
                                file["timeseries/t/" * string($iter)] / super_rotation_period)

    zonal_average_u = @lift dropdims(mean(file["timeseries/u/" * string($iter)][2:Nx+1, 2:Ny+1, 1], dims=1), dims=1)

    fig = Figure(resolution = (1080, 1080))

    ax = fig[1, 1] = Axis(fig, xlabel = "U(φ)", ylabel = "φ")

    theory = lines!(ax, solid_body_rotation.(φ), φ, color=:black)
    simulation = lines!(ax, zonal_average_u, φ, color=:blue)

    supertitle = fig[0, :] = Label(fig, plot_title, textsize=30)

    leg = Legend(fig, [theory, simulation], ["U cos(φ)", "Simulation"], markersize = 7,
                 halign = :right, valign = :top, bgcolor = :transparent)

    record(fig, "zonally_averaged_solid_body_rotation_Nx$Nx.mp4", iterations, framerate=30) do i
        @info "Animating iteration $i/$(iterations[end])..."
        iter[] = i
    end

    return nothing
end

function analyze_solid_body_rotation(filepath)

    @show output_prefix = basename(filepath)[1:end-5]

    file = jldopen(filepath)

    iterations = parse.(Int, keys(file["timeseries/t"]))

    Nx = file["grid/Nx"]
    Ny = file["grid/Ny"]

    grid = LatitudeLongitudeGrid(size = (Nx, Ny, 1),
                                 radius = 1,
                                 latitude = (-80, 80),
                                 longitude = (-180, 180),
                                 z = (-1, 0))

    super_rotation_period = 2π * grid.radius / U

    λ = xnodes(Face, grid)
    φ = ynodes(Center, grid)

    λ = repeat(reshape(λ, Nx, 1), 1, Ny)
    φ = repeat(reshape(φ, 1, Ny), Nx, 1)

    maximum_error = [maximum(abs, (file["timeseries/u/$i"][1:Nx, 1:Ny, 1] .- solid_body_rotation.(φ)) / U)
                     for i in iterations]

    return iterations, maximum_error
end

filepath = run_solid_body_rotation(Nx=180, Ny=60, super_rotations=0.5, advection_scheme=VectorInvariant(), prefix = "2ndorder")
plot_zonal_average_solid_body_rotation(filepath)
filepath = run_solid_body_rotation(Nx=180, Ny=60, super_rotations=0.5, advection_scheme=WENO5(vector_invariant=true), prefix = "weno")
plot_zonal_average_solid_body_rotation(filepath)
