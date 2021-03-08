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
using Oceananigans.Utils: prettytime, hours
using Oceananigans.OutputWriters: JLD2OutputWriter, TimeInterval, IterationInterval

using Statistics
using JLD2
using Printf
using GLMakie

Nx = 120
Ny = 60

# A spherical domain
grid = RegularLatitudeLongitudeGrid(size = (Nx, Ny, 1),
                                    longitude = (-30, 30),
                                    latitude = (-15, 45),
                                    z = (-4000, 0))

free_surface = ExplicitFreeSurface(gravitational_acceleration=0.1)

coriolis = HydrostaticSphericalCoriolis(scheme = VectorInvariantEnstrophyConserving())

surface_wind_stress_parameters = (τ₀ = 5e-4,
                                  Lϕ = grid.Ly,
                                  ϕ₀ = 15)

surface_wind_stress(λ, ϕ, t, p) = - p.τ₀ * cos(2π * (ϕ - p.ϕ₀) / p.Lϕ)

surface_wind_stress_bc = BoundaryCondition(Flux,
                                           surface_wind_stress,
                                           parameters = surface_wind_stress_parameters)

u_bcs = UVelocityBoundaryConditions(grid, top = surface_wind_stress_bc)
                                        
model = HydrostaticFreeSurfaceModel(grid = grid,
                                    architecture = CPU(),
                                    momentum_advection = VectorInvariant(),
                                    free_surface = free_surface,
                                    coriolis = coriolis,
                                    boundary_conditions = (u=u_bcs,),
                                    buoyancy = nothing,
                                    closure = HorizontallyCurvilinearAnisotropicDiffusivity(νh=1e2, κh=1e2))

g = model.free_surface.gravitational_acceleration

gravity_wave_speed = sqrt(g * grid.Lz) # hydrostatic (shallow water) gravity wave speed

# Time-scale for gravity wave propagation across the smallest grid cell
wave_propagation_time_scale = min(grid.radius * cosd(maximum(abs, grid.ϕᵃᶜᵃ)) * deg2rad(grid.Δλ),
                                  grid.radius * deg2rad(grid.Δϕ)) / gravity_wave_speed

simulation = Simulation(model,
                        Δt = 0.1wave_propagation_time_scale,
                        stop_iteration = 10000,
                        iteration_interval = 100,
                        progress = s -> @info "Time = $(s.model.clock.time) / $(s.stop_time)")
                                                         
output_fields = merge(model.velocities, (η=model.free_surface.η,))

output_prefix = "barotropic_gyre"

simulation.output_writers[:fields] = JLD2OutputWriter(model, output_fields,
                                                      schedule = IterationInterval(100),
                                                      prefix = output_prefix,
                                                      field_slicer = nothing,
                                                      force = true)

run!(simulation)

filepath = simulation.output_writers[:fields].filepath

file = jldopen(filepath)

iterations = parse.(Int, keys(file["timeseries/t"]))

λ = xnodes(Face, grid)
ϕ = ynodes(Center, grid)

λ = repeat(reshape(λ, Nx+1, 1), 1, Ny)
ϕ = repeat(reshape(ϕ, 1, Ny), Nx+1, 1)

λ_azimuthal = λ .+ 180  # Convert to λ ∈ [0°, 360°]
ϕ_azimuthal = 90 .- ϕ   # Convert to ϕ ∈ [0°, 180°] (0° at north pole)

iter = Node(0)

plot_title = @lift @sprintf("Barotropic gyre: iteration = %d", $iter)

u = @lift file["timeseries/u/" * string($iter)][:, :, 1]

# Plot on the unit sphere to align with the spherical wireframe.
x = @. cosd(λ_azimuthal) * sind(ϕ_azimuthal)
y = @. sind(λ_azimuthal) * sind(ϕ_azimuthal)
z = @. cosd(ϕ_azimuthal)

fig = Figure(resolution = (1080, 1080))

ax = fig[1, 1] = LScene(fig)
wireframe!(ax, Sphere(Point3f0(0), 0.99f0), show_axis=false)
surface!(ax, x, y, z, color=u, colormap=:balance) #, colorrange=(0.0, 0.02))
rotate_cam!(ax.scene, (3π/4, π/8, 0))
zoom!(ax.scene, (0, 0, 0), 2, false)

supertitle = fig[0, :] = Label(fig, plot_title, textsize=30)

record(fig, output_prefix * ".mp4", iterations, framerate=30) do i
    @info "Animating iteration $i/$(iterations[end])..."
    iter[] = i
end
