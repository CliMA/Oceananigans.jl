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
using GLMakie

Nx = 360
Ny = 360

# A spherical domain
grid = RegularLatitudeLongitudeGrid(size = (Nx, Ny, 1),
                                    longitude = (-30, 30),
                                    latitude = (15, 75),
                                    z = (-4000, 0))

free_surface = ExplicitFreeSurface(gravitational_acceleration=0.1)

coriolis = HydrostaticSphericalCoriolis(scheme = VectorInvariantEnstrophyConserving())

surface_wind_stress_parameters = (τ₀ = 1e-4,
                                  Lφ = grid.Ly,
                                  φ₀ = 15)

surface_wind_stress(λ, φ, t, p) = p.τ₀ * cos(2π * (φ - p.φ₀) / p.Lφ)

surface_wind_stress_bc = BoundaryCondition(Flux,
                                           surface_wind_stress,
                                           parameters = surface_wind_stress_parameters)

μ = 1 / 60days

u_bottom_drag(i, j, grid, clock, fields, μ) = @inbounds - μ * fields.u[i, j, 1]
v_bottom_drag(i, j, grid, clock, fields, μ) = @inbounds - μ * fields.v[i, j, 1]

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

#variable_horizontal_diffusivity =
#    HorizontallyCurvilinearAnisotropicDiffusivity(νh = (λ, φ, z, t) -> νh₀ * cosd(φ))

variable_horizontal_diffusivity = HorizontallyCurvilinearAnisotropicDiffusivity(νh = νh₀)

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

progress(s) = @info @sprintf("Time: %s, iteration: %d, max(u): %.2e m s⁻¹",
                             prettytime(s.model.clock.time),
                             s.model.clock.iteration,
                             maximum(s.model.velocities.u))

simulation = Simulation(model,
                        Δt = 0.2wave_propagation_time_scale,
                        stop_time = 3years,
                        iteration_interval = 100,
                        progress = progress)
                                                         
output_fields = merge(model.velocities, (η=model.free_surface.η,))

output_prefix = "barotropic_gyre_Nx$(grid.Nx)_Ny$(grid.Ny)"

simulation.output_writers[:fields] = JLD2OutputWriter(model, output_fields,
                                                      schedule = TimeInterval(1day),
                                                      prefix = output_prefix,
                                                      field_slicer = nothing,
                                                      force = true)

run!(simulation)

filepath = simulation.output_writers[:fields].filepath

file = jldopen(filepath)

iterations = parse.(Int, keys(file["timeseries/t"]))

λu = xnodes(Face, grid)
φu = ynodes(Center, grid)

λc = xnodes(Center, grid)
φc = ynodes(Center, grid)

λu = repeat(reshape(λu, Nx+1, 1), 1, Ny) 
φu = repeat(reshape(φu, 1, Ny), Nx+1, 1)

λc = repeat(reshape(λc, Nx, 1), 1, Ny) 
φc = repeat(reshape(φc, 1, Ny), Nx, 1)

λu_azimuthal = λu .+ 180  # Convert to λ ∈ [0°, 360°]
φu_azimuthal = 90 .- φu   # Convert to φ ∈ [0°, 180°] (0° at north pole)

λc_azimuthal = λc .+ 180  # Convert to λ ∈ [0°, 360°]
φc_azimuthal = 90 .- φc   # Convert to φ ∈ [0°, 180°] (0° at north pole)

iter = Node(0)

plot_title = @lift @sprintf("Barotropic gyre: time = %s", prettytime(file["timeseries/t/" * string($iter)]))

u = @lift file["timeseries/u/" * string($iter)][:, :, 1]
η = @lift file["timeseries/η/" * string($iter)][:, :, 1]

# Plot on the unit sphere to align with the spherical wireframe.
xu = @. cosd(λu_azimuthal) * sind(φu_azimuthal)
yu = @. sind(λu_azimuthal) * sind(φu_azimuthal)
zu = @. cosd(φu_azimuthal)

xc = @. cosd(λc_azimuthal) * sind(φc_azimuthal)
yc = @. sind(λc_azimuthal) * sind(φc_azimuthal)
zc = @. cosd(φc_azimuthal)

fig = Figure(resolution = (2160, 1540))

ax = fig[1, 1] = LScene(fig)
wireframe!(ax, Sphere(Point3f0(0), 0.99f0), show_axis=false)
surface!(ax, xu, yu, zu, color=u, colormap=:balance)
rotate_cam!(ax.scene, (3π/4, -π/8, 0))
zoom!(ax.scene, (0, 0, 0), 2, true)

ax = fig[1, 2] = LScene(fig)
wireframe!(ax, Sphere(Point3f0(0), 0.99f0), show_axis=false)
surface!(ax, xc, yc, zc, color=η, colormap=:balance)
rotate_cam!(ax.scene, (3π/4, -π/8, 0))
zoom!(ax.scene, (0, 0, 0), 2, true)

supertitle = fig[0, :] = Label(fig, plot_title, textsize=50)

record(fig, output_prefix * ".mp4", iterations, framerate=30) do i
    @info "Animating iteration $i/$(iterations[end])..."
    iter[] = i
end
