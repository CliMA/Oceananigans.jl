# # Solid body rotation of a meridional sector on the sphere
#
# This script implements the "Global Steady State Nonlinear Zonal Geostrophic Flow"
# validation experiment from
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
using Oceananigans.Coriolis: HydrostaticSphericalCoriolis, VectorInvariantEnergyConserving, VectorInvariantEnstrophyConserving
using Oceananigans.Models.HydrostaticFreeSurfaceModels: HydrostaticFreeSurfaceModel, VectorInvariant, ExplicitFreeSurface
using Oceananigans.Utils: prettytime
using Oceananigans.OutputWriters: JLD2OutputWriter, TimeInterval, IterationInterval

grid = RegularLatitudeLongitudeGrid(size = (90, 15, 1),
                                    radius = 1,
                                    latitude = (-60, 60),
                                    longitude = (-180, 180),
                                    z = (-1, 0))

coriolis = HydrostaticSphericalCoriolis(rotation_rate = 1,
                                        stencil = VectorInvariantEnstrophyConserving())

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    momentum_advection = VectorInvariant(),
                                    free_surface = ExplicitFreeSurface(gravitational_acceleration=1),
                                    coriolis = coriolis,
                                    tracers = :c,
                                    buoyancy = nothing,
                                    closure = nothing)

U = 0.1
g = model.free_surface.gravitational_acceleration
R = model.grid.radius
Ω = model.coriolis.rotation_rate

uᵢ(λ, ϕ, z) = U * cosd(ϕ)
ηᵢ(λ, ϕ, z) = (R * Ω * U + U^2 / 2) * sind(ϕ)^2 / g

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

# Calculate stop time based on the time scale for circumglobal tracer advection

circumfrence = 2π * grid.radius
circumvection_timescale = circumfrence / U

simulation = Simulation(model,
                        Δt = 0.1wave_propagation_time_scale,
                        stop_time = 32 * circumvection_timescale,
                        iteration_interval = 100,
                        progress = s -> @info "Time = $(s.model.clock.time) / $(s.stop_time)")
                                                         
output_fields = merge(model.velocities, model.tracers, (η=model.free_surface.η,))

output_prefix = "solid_body_rotation_Nx$(grid.Nx)"

simulation.output_writers[:fields] = JLD2OutputWriter(model, output_fields,
                                                      schedule = TimeInterval(circumvection_timescale / 20),
                                                      prefix = output_prefix,
                                                      force = true)

run!(simulation)

# ## Visualizing the results

using JLD2, Printf, Oceananigans.Grids, GLMakie
using Oceananigans.Utils: hours

λ, ϕ, r = nodes(model.free_surface.η, reshape=true)

λ = λ .+ 180  # Convert to λ ∈ [0°, 360°]
ϕ = 90 .- ϕ   # Convert to ϕ ∈ [0°, 180°] (0° at north pole)

filepath = output_prefix * ".jld2"
file = jldopen(filepath)

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
    rotate_cam!(ax.scene, (-π/4, π/8, 0))
    zoom!(ax.scene, (0, 0, 0), 2, false)
end

supertitle = fig[0, :] = Label(fig, plot_title, textsize=30)

record(fig, output_prefix * ".mp4", iterations, framerate=30) do i
    @info "Animating iteration $i/$(iterations[end])..."
    iter[] = i
end
