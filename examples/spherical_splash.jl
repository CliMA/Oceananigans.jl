# # Splash on a sphere using Oceananigans.HydrostaticFreeSurfaceModel
#
# ## Install dependencies
#
# First let's make sure we have all required packages installed.

# ```julia
# using Pkg
# pkg"add Oceananigans, JLD2, GLMakie"
# ```

# ## A spherical domain
#
# We two-dimensional latitude-longitude grid on a sphere of unit radius,

using Oceananigans
using Oceananigans.Grids: RegularLatitudeLongitudeGrid

grid = RegularLatitudeLongitudeGrid(size = (360, 60, 1),
                                    radius = 1,
                                    latitude = (-60, 60),
                                    longitude = (-180, 180),
                                    z = (-1, 0))

using Oceananigans.Coriolis: HydrostaticSphericalCoriolis

coriolis = HydrostaticSphericalCoriolis(rotation_rate=10.0)

# ## Building a `HydrostaticFreeSurfaceModel`
#
# We use `grid` and `coriolis` to build a simple `HydrostaticFreeSurfaceModel`,

using Oceananigans.Models.HydrostaticFreeSurfaceModels: HydrostaticFreeSurfaceModel
using Oceananigans.Models.HydrostaticFreeSurfaceModels: VectorInvariant, ExplicitFreeSurface

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    momentum_advection = VectorInvariant(),
                                    free_surface = ExplicitFreeSurface(gravitational_acceleration=1),
                                    tracers = (),
                                    buoyancy = nothing,
                                    coriolis = coriolis,
                                    closure = nothing)

# ## Making a splash
#
# We make a splash by imposing a Gaussian height field,

Gaussian(λ, φ, L) = exp(-(λ^2 + φ^2) / 2L^2)

a = 0.01 # splash amplitude
L = 10 # degree
φ₀ = 5 # degrees

splash(λ, φ) = a * Gaussian(λ, φ - φ₀, L)

set!(model, η=splash)

# ## Running a `Simulation`
#
# We pick a time-step that resolves the surface dynamics,

using Oceananigans.Utils: prettytime

g = model.free_surface.gravitational_acceleration

gravity_wave_speed = sqrt(g * grid.Lz) # hydrostatic (shallow water) gravity wave speed

# Time-scale for gravity wave propagation across the smallest grid cell
wave_propagation_time_scale = min(grid.radius * cosd(maximum(abs, grid.φᵃᶜᵃ)) * deg2rad(grid.Δλ),
                                  grid.radius * deg2rad(grid.Δφ)) / gravity_wave_speed

simulation = Simulation(model,
                        Δt = 0.05wave_propagation_time_scale,
                        stop_iteration = 4000,
                        iteration_interval = 100,
                        progress = s -> @info "Iteration = $(s.model.clock.iteration) / $(s.stop_iteration)")


# ## Output
#
# We output the velocity field and free surface displacement,

output_fields = merge(model.velocities, (η=model.free_surface.η,))

using Oceananigans.OutputWriters: JLD2OutputWriter, TimeInterval, IterationInterval

simulation.output_writers[:fields] = JLD2OutputWriter(model, output_fields,
                                                      schedule = IterationInterval(10),
                                                      prefix = "spherical_splash",
                                                      force = true)

run!(simulation)

# ## Visualizing the results

using JLD2, Printf, Oceananigans.Grids, GLMakie
using Oceananigans.Utils: hours

λ, φ, r = nodes(model.free_surface.η, reshape=true)

λ = λ .+ 180  # Convert to λ ∈ [0°, 360°]
φ = 90 .- φ   # Convert to φ ∈ [0°, 180°] (0° at north pole)

file = jldopen(simulation.output_writers[:fields].filepath)

iterations = parse.(Int, keys(file["timeseries/t"]))

iter = Node(0)
plot_title = @lift @sprintf("Oceananigans.jl on the sphere! Spherical splash: u, v, η @ time = %s",
                            prettytime(file["timeseries/t/" * string($iter)]))

u = @lift file["timeseries/u/" * string($iter)][:, :, 1]
v = @lift file["timeseries/v/" * string($iter)][:, :, 1]
η = @lift file["timeseries/η/" * string($iter)][:, :, 1]

# Plot on the unit sphere to align with the spherical wireframe.
x3 = @. cosd(λ) * sind(φ)
y3 = @. sind(λ) * sind(φ)
z3 = @. cosd(φ) * λ ./ λ

x = @lift (1 .+ 20 .* file["timeseries/η/" * string($iter)][:, :, 1]) .* x3[:, :, 1]
y = @lift (1 .+ 20 .* file["timeseries/η/" * string($iter)][:, :, 1]) .* y3[:, :, 1]
z = @lift (1 .+ 20 .* file["timeseries/η/" * string($iter)][:, :, 1]) .* z3[:, :, 1]

fig = Figure(resolution = (1920, 1080))

clims = [(-0.003, 0.003), (-0.003, 0.003), (-0.01, 0.01)]

for (n, var) in enumerate((u, v, η))
    ax = fig[1, n] = LScene(fig, title="$n")
    wireframe!(ax, Sphere(Point3f0(0), 0.98f0), show_axis=false)
    surface!(ax, x, y, z, color=var, colormap=:balance, colorrange=clims[n])
    rotate_cam!(ax.scene, (2π/3, 0, 0))
    zoom!(ax.scene, (0, 0, 0), 2, false)
end

supertitle = fig[0, :] = Label(fig, plot_title, textsize=30)

record(fig, "spherical_splash.mp4", iterations, framerate=30) do i
    @info "Animating iteration $i/$(iterations[end])..."
    iter[] = i
end
