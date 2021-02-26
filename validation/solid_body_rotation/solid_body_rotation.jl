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
using Oceananigans.Coriolis: HydrostaticSphericalCoriolis, VectorInvariantEnergyConserving, VectorInvariantEnstrophyConserving
using Oceananigans.Models.HydrostaticFreeSurfaceModels: HydrostaticFreeSurfaceModel, VectorInvariant, ExplicitFreeSurface
using Oceananigans.Utils: prettytime, hours
using Oceananigans.OutputWriters: JLD2OutputWriter, TimeInterval, IterationInterval

using JLD2
using Printf
using GLMakie

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

function run_solid_body_rotation(; architecture = CPU(),
                                   resolution = 4, # degrees
                                   coriolis_scheme = VectorInvariantEnstrophyConserving(),
                                   super_rotations = 4
                                )

    Nx = round(Int, 360 / resolution)
    Ny = round(Int, 120 / resolution)

    # A spherical domain
    grid = RegularLatitudeLongitudeGrid(size = (Nx, Ny, 1),
                                        radius = 1,
                                        latitude = (-60, 60),
                                        longitude = (-180, 180),
                                        z = (-1, 0))

    free_surface = ExplicitFreeSurface(gravitational_acceleration=1)

    coriolis = HydrostaticSphericalCoriolis(rotation_rate = 1,
                                            scheme = VectorInvariantEnstrophyConserving())

    model = HydrostaticFreeSurfaceModel(grid = grid,
                                        architecture = architecture,
                                        momentum_advection = VectorInvariant(),
                                        free_surface = free_surface,
                                        coriolis = coriolis,
                                        tracers = :c,
                                        buoyancy = nothing,
                                        closure = nothing)

    g = model.free_surface.gravitational_acceleration
    R = model.grid.radius
    Ω = model.coriolis.rotation_rate

    uᵢ(λ, ϕ, z) = solid_body_rotation(λ, ϕ, U)
    ηᵢ(λ, ϕ, z) = solid_body_geostrophic_height(λ, ϕ, U, R, Ω, g)

    # Tracer patch for visualization
    Gaussian(λ, ϕ, L) = exp(-(λ^2 + ϕ^2) / 2L^2)

    # Tracer patch parameters
    L = 10 # degree
    ϕ₀ = 5 # degrees

    cᵢ(λ, ϕ, z) = Gaussian(λ, ϕ - ϕ₀, L)

    set!(model, u=uᵢ, η=ηᵢ, c=cᵢ)

    gravity_wave_speed = sqrt(g * grid.Lz) # hydrostatic (shallow water) gravity wave speed

    # Time-scale for gravity wave propagation across the smallest grid cell
    wave_propagation_time_scale = min(grid.radius * cosd(maximum(abs, grid.ϕᵃᶜᵃ)) * deg2rad(grid.Δλ),
                                      grid.radius * deg2rad(grid.Δϕ)) / gravity_wave_speed

    super_rotation_period = 2π * grid.radius / U

    simulation = Simulation(model,
                            Δt = 0.1wave_propagation_time_scale,
                            stop_time = super_rotations * super_rotation_period,
                            iteration_interval = 100,
                            progress = s -> @info "Time = $(s.model.clock.time) / $(s.stop_time)")
                                                             
    output_fields = merge(model.velocities, model.tracers, (η=model.free_surface.η,))

    output_prefix = "solid_body_rotation_Nx$(grid.Nx)"

    simulation.output_writers[:fields] = JLD2OutputWriter(model, output_fields,
                                                          schedule = TimeInterval(super_rotation_period/ 20),
                                                          prefix = output_prefix,
                                                          force = true)

    run!(simulation)

    return simulation.output_writers[:fields].filepath
end

function visualize_solid_body_rotation(filepath)

    @show output_prefix = basename(filepath)[1:end-5]

    file = jldopen(filepath)

    iterations = parse.(Int, keys(file["timeseries/t"]))

    Nx = file["grid/Nx"]
    Ny = file["grid/Ny"]

    grid = RegularLatitudeLongitudeGrid(size = (Nx, Ny, 1),
                                        radius = 1,
                                        latitude = (-60, 60),
                                        longitude = (-180, 180),
                                        z = (-1, 0))

    super_rotation_period = 2π * grid.radius / U

    λ = xnodes(Face, grid)
    ϕ = ynodes(Center, grid)
    
    λ = repeat(reshape(λ, Nx, 1), 1, Ny)
    ϕ = repeat(reshape(ϕ, 1, Ny), Nx, 1)

    λ_azimuthal = λ .+ 180  # Convert to λ ∈ [0°, 360°]
    ϕ_azimuthal = 90 .- ϕ   # Convert to ϕ ∈ [0°, 180°] (0° at north pole)

    iter = Node(0)

    plot_title = @lift @sprintf("Zonal velocity error in solid body rotation: rotations = %.3f",
                                file["timeseries/t/" * string($iter)] / super_rotation_period)

    spatial_error = @lift abs.(file["timeseries/u/" * string($iter)][:, :, 1] .- solid_body_rotation.(λ, ϕ)) / U
    maximum_error = @lift maximum(abs, (file["timeseries/u/" * string($iter)][:, :, 1] .- solid_body_rotation.(λ, ϕ, U)) / U)

    # Plot on the unit sphere to align with the spherical wireframe.
    x = @. cosd(λ_azimuthal) * sind(ϕ_azimuthal)
    y = @. sind(λ_azimuthal) * sind(ϕ_azimuthal)
    z = @. cosd(ϕ_azimuthal)

    fig = Figure(resolution = (1080, 1080))

    ax = fig[1, 1] = LScene(fig, title="")
    wireframe!(ax, Sphere(Point3f0(0), 0.99f0), show_axis=false)
    surface!(ax, x, y, z, color=spatial_error, colormap=:thermal, colorrange=(0.0, 0.02))
    rotate_cam!(ax.scene, (-π/4, π/8, 0))
    zoom!(ax.scene, (0, 0, 0), 2, false)

    supertitle = fig[0, :] = Label(fig, plot_title, textsize=30)

    record(fig, output_prefix * ".mp4", iterations, framerate=30) do i
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

    grid = RegularLatitudeLongitudeGrid(size = (Nx, Ny, 1),
                                        radius = 1,
                                        latitude = (-60, 60),
                                        longitude = (-180, 180),
                                        z = (-1, 0))

    super_rotation_period = 2π * grid.radius / U

    λ = xnodes(Face, grid)
    ϕ = ynodes(Center, grid)
    
    λ = repeat(reshape(λ, Nx, 1), 1, Ny)
    ϕ = repeat(reshape(ϕ, 1, Ny), Nx, 1)

    maximum_error = [maximum(abs, (file["timeseries/u/$i"][:, :, 1] .- solid_body_rotation.(λ, ϕ, U)) / U)
                     for i in iterations]

    return iterations, maximum_error
end

for resolution in (1, 2, 4, 8) # degrees
    filepath = run_solid_body_rotation(resolution=resolution)
    visualize_solid_body_rotation(filepath)
end

#=
fig = Figure(resolution = (1080, 1080))
ax = fig[1, 1] = LScene(fig, title="")

for Nx in (45, 90) #, 180, 360)
    filepath = "solid_body_rotation_Nx$Nx.jld2"
    iterations, maximum_error = analyze_solid_body_rotation(filepath)
    lines!(ax, iterations, maximum_error)
end
=#
