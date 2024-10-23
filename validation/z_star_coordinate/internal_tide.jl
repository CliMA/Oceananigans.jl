using CairoMakie
using Oceananigans
using Oceananigans.Units
using Oceananigans.ImmersedBoundaries: PartialCellBottom
using Oceananigans.AbstractOperations: GridMetricOperation

Nx, Nz = 250, 125

H = 2kilometers
z_faces = ZStarVerticalCoordinate(-H, 0)

underlying_grid = RectilinearGrid(size = (Nx, Nz),
                                  x = (-1000kilometers, 1000kilometers),
                                  z = z_faces,
                                  halo = (4, 4),
                                  topology = (Periodic, Flat, Bounded))

h₀ = 250meters
width = 20kilometers
hill(x) = h₀ * exp(-x^2 / 2width^2)
bottom(x) = - H + hill(x)

grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(bottom))

coriolis = FPlane(latitude = -45)

# Now we have everything we require to construct the tidal forcing given a value of the
# excursion parameter.

T₂ = 12.421hours
ω₂ = 2π / T₂ # radians/sec

ϵ = 0.1 # excursion parameter

U_tidal = ϵ * ω₂ * width

tidal_forcing_amplitude = U_tidal * (ω₂^2 - coriolis.f^2) / ω₂

@inline tidal_forcing(x, z, t, p) = p.tidal_forcing_amplitude * sin(p.ω₂ * t)

u_forcing = Forcing(tidal_forcing, parameters=(; tidal_forcing_amplitude, ω₂))

# ## Model

# We built a `HydrostaticFreeSurfaceModel`:

model = HydrostaticFreeSurfaceModel(; grid, coriolis,
                                      buoyancy = BuoyancyTracer(),
                                      tracers = :b,
                                      free_surface = SplitExplicitFreeSurface(grid; substeps = 20),
                                      momentum_advection = WENO(),
                                      tracer_advection = WENO(),
                                      forcing = (; u = u_forcing))

# We initialize the model with the tidal flow and a linear stratification.

uᵢ(x, z) = 0

Nᵢ² = 1e-4  # [s⁻²] initial buoyancy frequency / stratification
bᵢ(x, z) = Nᵢ² * z

set!(model, u=uᵢ, b=bᵢ)

# Now let's build a `Simulation`.

Δt = 5minutes
stop_time = 4days

simulation = Simulation(model; Δt, stop_time, stop_iteration = 2)

# We add a callback to print a message about how the simulation is going,

using Printf

wall_clock = Ref(time_ns())

dz = GridMetricOperation((Center, Center, Center), Oceananigans.AbstractOperations.Δz, model.grid)
∫b_init = sum(model.tracers.b * dz) / sum(dz)

function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])

    ∫b = sum(model.tracers.b * dz) / sum(dz)

    msg = @sprintf("iteration: %d, time: %s, wall time: %s, max|w|: %6.3e, max|u|: %6.3e, drift: %6.3e\n",
                   iteration(sim), prettytime(sim), prettytime(elapsed),
                   maximum(abs, interior(w, :, :, grid.Nz+1)), maximum(abs, simulation.model.velocities.u),
                   ∫b - ∫b_init)

    wall_clock[] = time_ns()

    @info msg

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

b = model.tracers.b
u, v, w = model.velocities

U = Field(Average(u))

u′ = u - U

N² = ∂z(b)

filename = "internal_tide"
save_fields_interval = 30minutes

simulation.output_writers[:fields] = JLD2OutputWriter(model, (; u, u′, w, b, N²);
                                                      filename,
                                                      schedule = TimeInterval(save_fields_interval),
                                                      overwrite_existing = true)

# We are ready -- let's run!
run!(simulation)

# # ## Load output

# # # First, we load the saved velocities and stratification output as `FieldTimeSeries`es.

# saved_output_filename = filename * ".jld2"

# u′_t = FieldTimeSeries(saved_output_filename, "u′")
#  w_t = FieldTimeSeries(saved_output_filename, "w")
# N²_t = FieldTimeSeries(saved_output_filename, "N²")

# umax = maximum(abs, u′_t[end])
# wmax = maximum(abs, w_t[end])

# times = u′_t.times
# nothing #hide

# # We retrieve each field's coordinates and convert from meters to kilometers.

# xu,  _, zu  = nodes(u′_t[1])
# xw,  _, zw  = nodes(w_t[1])
# xN², _, zN² = nodes(N²_t[1])

# xu  = xu  ./ 1e3
# xw  = xw  ./ 1e3
# xN² = xN² ./ 1e3
# zu  = zu  ./ 1e3
# zw  = zw  ./ 1e3
# zN² = zN² ./ 1e3
# nothing #hide

# # ## Visualize

# # Now we can visualize our resutls! We use `CairoMakie` here. On a system with OpenGL
# # `using GLMakie` is more convenient as figures will be displayed on the screen.
# #
# # We use Makie's `Observable` to animate the data. To dive into how `Observable`s work we
# # refer to [Makie.jl's Documentation](https://makie.juliaplots.org/stable/documentation/nodes/index.html).

# using CairoMakie

# n = Observable(1)

# title = @lift @sprintf("t = %1.2f days = %1.2f T₂",
#                        round(times[$n] / day, digits=2) , round(times[$n] / T₂, digits=2))

# u′n = @lift u′_t[$n]
#  wn = @lift  w_t[$n]
# N²n = @lift N²_t[$n]

# axis_kwargs = (xlabel = "x [km]",
#                ylabel = "z [km]",
#                limits = ((-grid.Lx/2e3, grid.Lx/2e3), (-grid.Lz/1e3, 0)), # note conversion to kilometers
#                titlesize = 20)

# fig = Figure(size = (700, 900))

# fig[1, :] = Label(fig, title, fontsize=24, tellwidth=false)

# ax_u = Axis(fig[2, 1]; title = "u'-velocity", axis_kwargs...)
# hm_u = heatmap!(ax_u, xu, zu, u′n; nan_color=:gray, colorrange=(-umax, umax), colormap=:balance)
# Colorbar(fig[2, 2], hm_u, label = "m s⁻¹")

# ax_w = Axis(fig[3, 1]; title = "w-velocity", axis_kwargs...)
# hm_w = heatmap!(ax_w, xw, zw, wn; nan_color=:gray, colorrange=(-wmax, wmax), colormap=:balance)
# Colorbar(fig[3, 2], hm_w, label = "m s⁻¹")

# ax_N² = Axis(fig[4, 1]; title = "stratification N²", axis_kwargs...)
# hm_N² = heatmap!(ax_N², xN², zN², N²n; nan_color=:gray, colorrange=(0.9Nᵢ², 1.1Nᵢ²), colormap=:magma)
# Colorbar(fig[4, 2], hm_N², label = "s⁻²")

# fig

# # Finally, we can record a movie.

# @info "Making an animation from saved data..."

# frames = 1:length(times)

# record(fig, filename * ".mp4", frames, framerate=16) do i
#     @info string("Plotting frame ", i, " of ", frames[end])
#     n[] = i
# end
# nothing #hide

# # ![](internal_tide.mp4)
