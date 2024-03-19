# # Internal tide by a seamount
#
# In this example, we show how internal tide is generated from a barotropic tidal flow
# sloshing back and forth over a sea mount.
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

Nx, Nz = 250, 125

H = 2kilometers

underlying_grid = RectilinearGrid(size = (Nx, Nz),
                                  x = (-1000kilometers, 1000kilometers),
                                  z = (-H, 0),
                                  halo = (4, 4),
                                  topology = (Periodic, Flat, Bounded))

# Now we can create the non-trivial bathymetry. We use `GridFittedBottom` that gets as input either
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
width = 20kilometers
hill(x) = h₀ * exp(-x^2 / 2width^2)
bottom(x) = - H + hill(x)

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom))

# Let's see how the domain with the bathymetry is.

x = xnodes(grid, Center())
bottom_boundary = interior(grid.immersed_boundary.bottom_height, :, 1, 1)
top_boundary = 0*x

using CairoMakie

fig = Figure(size = (700, 200))
ax = Axis(fig[1, 1],
          xlabel="x [km]",
          ylabel="z [m]",
          limits=((-grid.Lx/2e3, grid.Lx/2e3), (-grid.Lz, 0)))

band!(ax, x/1e3, bottom_boundary, top_boundary, color = :mediumblue)

fig

# Now we want to add a barotropic tide forcing. For example, to add the lunar semi-diurnal ``M_2`` tide 
# we need to add forcing in the ``u``-momentum equation of the form:
# ```math
# F_0 \sin(\omega_2 t)
# ```
# where ``\omega_2 = 2π / T_2``, with ``T_2 = 12.421 \,\mathrm{hours}`` the period of the ``M_2`` tide.

# The excursion parameter is a nondimensional number that expresses the ratio of the flow movement
# due to the tide compared to the size of the width of the hill.
#
# ```math
# \epsilon = \frac{U_{\mathrm{tidal}} / \omega_2}{\sigma}
# ```
# 
# We prescribe the excursion parameter which, in turn, implies a tidal velocity ``U_{\mathrm{tidal}}``
# which then allows us to determing the tidal forcing amplitude ``F_0``. For the last step, we
# use Fourier decomposition on the inviscid, linearized momentum equations to determine the
# flow response for a given tidal forcing. Doing so we get that for the sinusoidal forcing above,
# the tidal velocity and tidal forcing amplitudes are related via:
#
# ```math
# U_{\mathrm{tidal}} = \frac{\omega_2}{\omega_2^2 - f^2} F_0
# ```
#
# Now we have the way to find the value of the tidal forcing amplitude that would correspond to a
# given excursion parameter. The Coriolis frequency is needed, so we start by constructing a Coriolis on an ``f``-plane at the
# mid-latitudes.

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
                                      momentum_advection = WENO(),
                                      tracer_advection = WENO(),
                                      forcing = (; u = u_forcing))

# We initialize the model with the tidal flow and a linear stratification.

uᵢ(x, z) = U_tidal

Nᵢ² = 1e-4  # [s⁻²] initial buoyancy frequency / stratification
bᵢ(x, z) = Nᵢ² * z

set!(model, u=uᵢ, b=bᵢ)

# Now let's built a `Simulation`.

Δt = 5minutes
stop_time = 4days

simulation = Simulation(model; Δt, stop_time)

# We add a callback to print a message about how the simulation is going,

using Printf

wall_clock = Ref(time_ns())

function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])

    msg = @sprintf("iteration: %d, time: %s, wall time: %s, max|w|: %6.3e, m s⁻¹\n",
                   iteration(sim), prettytime(sim), prettytime(elapsed),
                   maximum(abs, sim.model.velocities.w))

    wall_clock[] = time_ns()

    @info msg

    return nothing
end

add_callback!(simulation, progress, name=:progress, IterationInterval(200))
nothing #hide

# ## Diagnostics/Output

# Add some diagnostics. Instead of ``u`` we save the deviation of ``u`` from its instantaneous
# domain average, ``u' = u - (L_x H)^{-1} \int u \, \mathrm{d}x \mathrm{d}z``. We also save
# the stratification ``N^2 = \partial_z b``.

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

# ## Load output

# First, we load the saved velocities and stratification output as `FieldTimeSeries`es.

saved_output_filename = filename * ".jld2"

u′_t = FieldTimeSeries(saved_output_filename, "u′")
 w_t = FieldTimeSeries(saved_output_filename, "w")
N²_t = FieldTimeSeries(saved_output_filename, "N²")

umax = maximum(abs, u′_t[end])
wmax = maximum(abs, w_t[end])

times = u′_t.times
nothing #hide

# For visualization purposes, we mask the region below the bathymetry with NaNs.

using Oceananigans.ImmersedBoundaries: mask_immersed_field!

for φ_t in (u′_t, w_t, N²_t), n in 1:length(times)
    mask_immersed_field!(φ_t[n], NaN)
end

# We retrieve each field's coordinates and convert from meters to kilometers.

xu,  yu,  zu  = nodes(u′_t[1]) ./ 1e3
xw,  yw,  zw  = nodes(w_t[1])  ./ 1e3
xN², yN², zN² = nodes(N²_t[1]) ./ 1e3
nothing #hide

# ## Visualize

# Now we can visualize our resutls! We use `CairoMakie` here. On a system with OpenGL
# `using GLMakie` is more convenient as figures will be displayed on the screen.
#
# We use Makie's `Observable` to animate the data. To dive into how `Observable`s work we
# refer to [Makie.jl's Documentation](https://makie.juliaplots.org/stable/documentation/nodes/index.html).

using CairoMakie

n = Observable(1)

title = @lift @sprintf("t = %1.2f days = %1.2f T₂",
                       round(times[$n] / day, digits=2) , round(times[$n] / T₂, digits=2))

u′ₙ = @lift interior(u′_t[$n], :, 1, :)
 wₙ = @lift interior( w_t[$n], :, 1, :)
N²ₙ = @lift interior(N²_t[$n], :, 1, :)

axis_kwargs = (xlabel = "x [km]",
               ylabel = "z [km]",
               limits = ((-grid.Lx/2e3, grid.Lx/2e3), (-grid.Lz/1e3, 0)), # note conversion to kilometers
               titlesize = 20)

fig = Figure(size = (700, 900))

fig[1, :] = Label(fig, title, fontsize=24, tellwidth=false)

ax_u = Axis(fig[2, 1]; title = "u'-velocity", axis_kwargs...)
hm_u = heatmap!(ax_u, xu, zu, u′ₙ; colorrange = (-umax, umax), colormap = :balance)
Colorbar(fig[2, 2], hm_u, label = "m s⁻¹")

ax_w = Axis(fig[3, 1]; title = "w-velocity", axis_kwargs...)
hm_w = heatmap!(ax_w, xw, zw, wₙ; colorrange = (-wmax, wmax), colormap = :balance)
Colorbar(fig[3, 2], hm_w, label = "m s⁻¹")

ax_N² = Axis(fig[4, 1]; title = "stratification N²", axis_kwargs...)
hm_N² = heatmap!(ax_N², xN², zN², N²ₙ; colorrange = (0.9Nᵢ², 1.1Nᵢ²), colormap = :thermal)
Colorbar(fig[4, 2], hm_N², label = "s⁻²")

fig

# Finally, we can record a movie.

@info "Making an animation from saved data..."

frames = 1:length(times)

record(fig, filename * ".mp4", frames, framerate=16) do i
    @info string("Plotting frame ", i, " of ", frames[end])
    n[] = i
end
nothing #hide

# ![](internal_tide.mp4)
