# # Compact zonal geostrophic flow on the sphere
#
# This script implements the "Steady State Nonlinear Zonal Geostrophic Flow with
# Compact Support" validation experiment from
#
# > Williamson et al., "A Standard Test Set for Numerical Approximations to the Shallow
#   Water Equations in Spherical Geometry", Journal of Computational Physics, 1992.
#
# The geostrophic flow
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
#
# ## A spherical domain
#
# We two-dimensional latitude-longitude grid on a sphere of unit radius,

using Oceananigans
using Oceananigans.Grids: RegularLatitudeLongitudeGrid
using Oceananigans.Coriolis: HydrostaticSphericalCoriolis
using Oceananigans.Models.HydrostaticFreeSurfaceModels: HydrostaticFreeSurfaceModel, VectorInvariant, ExplicitFreeSurface
using Oceananigans.Utils: prettytime
using Oceananigans.OutputWriters: JLD2OutputWriter, TimeInterval, IterationInterval

grid = RegularLatitudeLongitudeGrid(size = (360, 60, 1),
                                    radius = 1,
                                    latitude = (-60, 60),
                                    longitude = (-180, 180),
                                    z = (-1, 0))

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    momentum_advection = VectorInvariant(),
                                    free_surface = ExplicitFreeSurface(gravitational_acceleration=1),
                                    tracers = :c,
                                    buoyancy = nothing,
                                    coriolis = HydrostaticSphericalCoriolis(rotation_rate=1),
                                    closure = nothing)

U = 0.1
g = model.free_surface.gravitational_acceleration
R = model.grid.radius
Ω = model.coriolis.rotation_rate

ϕᵇ = deg2rad(-π/6)
ϕᵉ = deg2rad(+π/2)
ξᵉ = 0.3

ξ(ϕ) = ξᵉ * (ϕ - ϕᵇ) / (ϕᵉ - ϕᵇ)

uᵢ(λ, ϕ, z) = U * b(ξ(ϕ)) * b(ξᵉ - ξ(ϕ)) * exp(4 / ξᵉ)
vᵢ(λ, ϕ, z) = 0

# Tracer patch for visualization
Gaussian(λ, ϕ, L) = exp(-(λ^2 + ϕ^2) / 2L^2)

# Tracer patch parameters
L = 10 # degree
ϕ₀ = 5 # degrees

cᵢ(λ, ϕ, z) = Gaussian(λ, ϕ - ϕ₀, L)

set!(model, u=uᵢ, v=vᵢ, c=cᵢ)

# Integrate to find ``η``

gravity_wave_speed = sqrt(g * grid.Lz) # hydrostatic (shallow water) gravity wave speed

# Time-scale for gravity wave propagation across the smallest grid cell
wave_propagation_time_scale = min(grid.radius * cosd(maximum(abs, grid.ϕᵃᶜᵃ)) * deg2rad(grid.Δλ),
                                  grid.radius * deg2rad(grid.Δϕ)) / gravity_wave_speed

circumfrence = 2π * grid.radius
circumvection_timescale = circumfrence / U

simulation = Simulation(model,
                        Δt = 0.1wave_propagation_time_scale,
                        stop_iteration = 1.5 * circumvection_timescale,
                        iteration_interval = 100,
                        progress = s -> @info "Iteration = $(s.model.clock.iteration) / $(s.stop_iteration)")
                                                         
output_fields = merge(model.velocities, model.tracers, (η=model.free_surface.η,))

simulation.output_writers[:fields] = JLD2OutputWriter(model, output_fields,
                                                      schedule = TimeInterval(circumvection_timescale / 20),
                                                      prefix = "solid_body_rotation_Nx$(grid.Nx)",
                                                      force = true)

run!(simulation)

# ## Visualizing the results

using JLD2, Printf, Oceananigans.Grids, GLMakie
using Oceananigans.Utils: hours

λ, ϕ, r = nodes(model.free_surface.η, reshape=true)

λ = λ .+ 180  # Convert to λ ∈ [0°, 360°]
ϕ = 90 .- ϕ   # Convert to ϕ ∈ [0°, 180°] (0° at north pole)

file = jldopen(simulation.output_writers[:fields].filepath)

iterations = parse.(Int, keys(file["timeseries/t"]))

iter = Node(0)
plot_title = @lift @sprintf("Tracer advection by solid body rotation (u, v, η, c): time = %s",
                            prettytime(file["timeseries/t/" * string($iter)]))

u = @lift file["timeseries/u/" * string($iter)][:, :, 1]
v = @lift file["timeseries/v/" * string($iter)][:, :, 1]
η = @lift file["timeseries/η/" * string($iter)][:, :, 1]
c = @lift file["timeseries/c/" * string($iter)][:, :, 1]

# Plot on the unit sphere to align with the spherical wireframe.
x3 = @. cosd(λ) * sind(ϕ)
y3 = @. sind(λ) * sind(ϕ)
z3 = @. cosd(ϕ) * λ ./ λ

x = x3[:, :, 1]
y = y3[:, :, 1]
z = z3[:, :, 1]

fig = Figure(resolution = (2400, 1080))

clims = [(-U, U), (-U, U), (-U, U), (0, 1)]

for (n, var) in enumerate((u, v, η, c))
    ax = fig[1, n] = LScene(fig, title="hi")
    wireframe!(ax, Sphere(Point3f0(0), 0.98f0), show_axis=false) #, color=RGBA(0.0, 0.0, 0.2, 0.4))
    surface!(ax, x, y, z, color=var, colormap=:balance, colorrange=clims[n])
    rotate_cam!(ax.scene, (2π/3, 0, 0))
    zoom!(ax.scene, (0, 0, 0), 2, false)
end

supertitle = fig[0, :] = Label(fig, plot_title, textsize=30)

record(fig, "solid_body_rotation_Nx$(grid.Nx).mp4", iterations, framerate=30) do i
    @info "Animating iteration $i/$(iterations[end])..."
    iter[] = i
end
