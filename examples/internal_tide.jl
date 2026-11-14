# # Internal tide over a seamount
#
# In this example, we show how an internal tide is generated from a barotropic tidal flow
# sloshing back and forth over a seamount.
#
# ## Install dependencies
#
# First let's make sure we have all required packages installed.

# ```julia
# using Pkg
# pkg"add Oceananigans, CairoMakie"
# ```

using Oceananigans
using Oceananigans.Units

# ## Grid

# We create an `ImmersedBoundaryGrid` wrapped around an underlying two-dimensional `RectilinearGrid`
# that is periodic in ``x`` and bounded in ``z``.

Nx, Nz = 256, 128
H, L = 2kilometers, 1000kilometers

underlying_grid = RectilinearGrid(size = (Nx, Nz), halo = (4, 4),
                                  x = (-L, L), z = (-H, 0),
                                  topology = (Periodic, Flat, Bounded))

# Now we can create the non-trivial bathymetry. We use `PartialCellBottom` that gets as input either
# *(i)* a two-dimensional function whose arguments are the grid's native horizontal coordinates and
# it returns the ``z`` of the bottom, or *(ii)* a two-dimensional array with the values of ``z`` at
# the bottom cell centers.
#
# In this example we'd like to have a Gaussian hill at the center of the domain.
#
# ```math
# h(x) = -H + h_0 \exp(-x^2 / 2σ^2)
# ```

h₀ = 250meters
σ = 20kilometers
bottom(x) = -H + h₀ * exp(-x^2 / 2σ^2)

grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(bottom))

# Let's see what the domain with the bathymetry looks like.

x = xnodes(grid, Center())
bottom_height = interior(bottom_height_field(grid), :, 1, 1)

using CairoMakie

fig = Figure(size = (700, 200))
ax = Axis(fig[1, 1],
          xlabel = "x [km]",
          ylabel = "z [m]",
          limits = ((-L, L) ./ kilometers, (-H, 0)))

band!(ax, x / kilometers, bottom_height, zero(x), color = :mediumblue)

fig

# Now we want to add a barotropic tide forcing. For example, to add the lunar semi-diurnal ``M_2`` tide
# we need to add forcing in the ``u``-momentum equation of the form:
# ```math
# F_0 \sin(ω_2 t)
# ```
# where ``ω_2 = 2π / T_2``, with ``T_2 = 12.421 \,\mathrm{hours}`` the period of the ``M_2`` tide.

# The excursion parameter is a nondimensional number that expresses the ratio of the
# tidal excursion to the width of the hill,
#
# ```math
# ϵ = \frac{U_2 / ω_2}{σ}
# ```
#
# We prescribe the excursion parameter which, in turn, implies a tidal velocity ``U_2``
# which then allows us to determine the tidal forcing amplitude ``F_0``. For the last step, we
# use Fourier decomposition on the inviscid, linearized momentum equations to determine the
# flow response for a given tidal forcing. Doing so we get that for the sinusoidal forcing above,
# the tidal velocity and tidal forcing amplitudes are related via:
#
# ```math
# U_2 = \frac{ω_2}{ω_2^2 - f^2} F_0
# ```
#
# Now we have a way to find the tidal forcing amplitude that corresponds to a given
# excursion parameter. The Coriolis frequency is needed, so we start by constructing
# an ``f``-plane at mid-latitudes.

coriolis = FPlane(latitude = -45)
f = coriolis.f

# Now we have everything we require to construct the tidal forcing given a value of the
# excursion parameter.

T₂ = 12.421hours
ω₂ = 2π / T₂
ϵ = 0.1
U₂ = ϵ * ω₂ * σ
F₀ = U₂ * (ω₂^2 - f^2) / ω₂

@inline tidal_forcing(x, z, t, p) = p.F₀ * sin(p.ω₂ * t)
u_forcing = Forcing(tidal_forcing, parameters=(; F₀, ω₂))

# ## Model

# We build a `HydrostaticFreeSurfaceModel`:

model = HydrostaticFreeSurfaceModel(grid; coriolis,
                                    buoyancy = BuoyancyTracer(),
                                    tracers = :b,
                                    momentum_advection = WENO(),
                                    tracer_advection = WENO(),
                                    forcing = (; u = u_forcing))

# We initialize the model with the tidal flow and a linear stratification.

Nᵢ² = 1e-4  # s⁻²
bᵢ(x, z) = Nᵢ² * z
set!(model, u=U₂, b=bᵢ)

# Now let's build a `Simulation`.

simulation = Simulation(model, Δt=5minutes, stop_time=4days)

# We add a callback to print a message about how the simulation is going,

using Printf

progress(sim) = @info @sprintf("Iter: %d, time: %s, wall time: %s, max|w|: %6.3e m s⁻¹",
                               iteration(sim), prettytime(sim), prettytime(sim.run_wall_time),
                               maximum(abs, sim.model.velocities.w))

add_callback!(simulation, progress, IterationInterval(200))
nothing #hide

# ## Diagnostics/Output

# Add some diagnostics. Instead of ``u`` we save the deviation of ``u`` from its instantaneous
# domain average, ``u′ = u - (L_x H)^{-1} \int u \, \mathrm{d}x \mathrm{d}z``. We also save
# the stratification ``N^2 = ∂_z b``.

b = model.tracers.b
u, v, w = model.velocities
U = Field(Average(u))
u′ = u - U
N² = ∂z(b)

filename = "internal_tide.jld2"

simulation.output_writers[:fields] = JLD2Writer(model, (; u, u′, w, b, N²); filename,
                                                schedule = TimeInterval(30minutes),
                                                overwrite_files = true)

# We are ready -- let's run!

## Fail the docs build if this simulation produces NaNs #hide
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation) #hide
run!(simulation)

# ## Load output

# First, we load the saved velocities and stratification output as `FieldTimeSeries`es.

u′_timeseries = FieldTimeSeries(filename, "u′")
 w_timeseries = FieldTimeSeries(filename, "w")
N²_timeseries = FieldTimeSeries(filename, "N²")

u′max = maximum(abs, u′_timeseries[end])
 wmax = maximum(abs, w_timeseries[end])

times = u′_timeseries.times
nothing #hide

# ## Visualize

# Now we can visualize our results! We use `CairoMakie` here. On a system with OpenGL
# `using GLMakie` is more convenient as figures will be displayed on the screen.
#
# We use Makie's `Observable` to animate the data. To dive into how `Observable`s work we
# refer to [Makie.jl's Documentation](https://docs.makie.org/stable/explanations/observables).

n = Observable(1)

title = @lift @sprintf("t = %1.2f days = %1.2f T₂", times[$n] / day, times[$n] / T₂)

u′ₙ = @lift u′_timeseries[$n]
 wₙ = @lift w_timeseries[$n]
N²ₙ = @lift N²_timeseries[$n]

axis_kwargs = (xlabel = "x [m]",
               ylabel = "z [m]",
               limits = ((-L, L), (-H, 0)),
               titlesize = 20)

fig = Figure(size = (700, 900))

fig[1, :] = Label(fig, title, fontsize=24, tellwidth=false)

ax_u = Axis(fig[2, 1]; title = "u'-velocity", axis_kwargs...)
hm_u = heatmap!(ax_u, u′ₙ; nan_color=:gray, colorrange=(-u′max, u′max), colormap=:balance)
Colorbar(fig[2, 2], hm_u, label = "m s⁻¹")

ax_w = Axis(fig[3, 1]; title = "w-velocity", axis_kwargs...)
hm_w = heatmap!(ax_w, wₙ; nan_color=:gray, colorrange=(-wmax, wmax), colormap=:balance)
Colorbar(fig[3, 2], hm_w, label = "m s⁻¹")

ax_N² = Axis(fig[4, 1]; title = "stratification N²", axis_kwargs...)
hm_N² = heatmap!(ax_N², N²ₙ; nan_color=:gray, colorrange=(0.9Nᵢ², 1.1Nᵢ²), colormap=:magma)
Colorbar(fig[4, 2], hm_N², label = "s⁻²")

fig

# Finally, we can record a movie.

@info "Making an animation from saved data..."

frames = 1:length(times)

record(fig, "internal_tide.mp4", frames, framerate=16) do i
    n[] = i
end
nothing #hide

# ![](internal_tide.mp4)
