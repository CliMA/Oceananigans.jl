# # Barotropic tide
#
# In this example, we simulate the evolution of a barotropic tide over a hill.
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

# We use an immersed boundary two-dimensional grid (in ``x``--``z``) that is is periodic in
# the ``x``-direction. To construct an immersed boundary grid we first need to create what
# we refer to as "underlying grid", which the grid that encompasses the immersed boundary.

Nx, Nz = 200, 60

H  = 2kilometers
Lx = 2000kilometers

underlying_grid = RectilinearGrid(size = (Nx, Nz),
                                  x = (-Lx/2, Lx/2),
                                  z = (-H, 0),
                                  halo = (4, 4),
                                  topology = (Periodic, Flat, Bounded))

# Now we want to create a bathymetry as an immersed boundary grid.

# ```math
# h(x) = -H + h_0 \exp(-x^2 / 2σ^2)
# ```

h₀ = 50 # m
width = 5kilometers
bump(x, y) = - H + h₀ * exp(-x^2 / 2width^2)

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bump))

# The hill is small; here's how it looks (note that we don't plot all the way to the ocean surface).

xC = xnodes(grid, Center())
bottom = grid.immersed_boundary.bottom_height[1:Nx, 1]

using CairoMakie

fig = Figure(resolution = (700, 200))
ax = Axis(fig[1, 1],
          xlabel="x [km]",
          ylabel="z [m]",
          limits=((-Lx/2e3, Lx/2e3), (-H, -4H/5)))

lines!(ax, xC/1e3, bottom)

fig

# Now we want to add a barotropic tide forcing. For example, to add the ``M_2`` tidal forcing
# we need to add forcing in the ``u``-momentum equation of the form:
# ```math
# \partial_t u = \dotsb + f_0 \sin(\omega_2 t)
# ```
# where ``\omega_2 = 2π / T_2``, with ``T_2`` the period of the ``M_2`` tide.

T₂ = 12.421hours
const ω₂ = 2π / T₂ # radians/sec

coriolis = FPlane(latitude = -45)

ε = 0.25

U_tidal = ε * ω₂ * width

const tidal_forcing_amplitude = U_tidal * (coriolis.f^2 - ω₂^2) / ω₂

@inline tidal_forcing(x, y, z, t) = tidal_forcing_amplitude * sin(ω₂ * t)


# ## Model

# We built a `HydrostaticFreeSurfaceModel` with an `ImplicitFreeSurface` solver.

using Oceananigans.Models.HydrostaticFreeSurfaceModels: FFTImplicitFreeSurfaceSolver

fft_preconditioner = FFTImplicitFreeSurfaceSolver(grid)
free_surface = ImplicitFreeSurface(solver_method=:PreconditionedConjugateGradient, preconditioner=fft_preconditioner)

model = HydrostaticFreeSurfaceModel(; grid, free_surface, coriolis,
                                      buoyancy = BuoyancyTracer(),
                                      tracers = :b,
                                      momentum_advection = WENO(),
                                      tracer_advection = WENO(),
                                      forcing = (u = tidal_forcing,))

# We initialize the model with the tidal flow and a linear stratification.

uᵢ(x, y, z) = U_tidal

Nᵢ² = 2e-4  # [s⁻²] initial buoyancy frequency / stratification
bᵢ(x, y, z) = Nᵢ² * z

set!(model, u=uᵢ, b=bᵢ)

# Now let's built a `Simulation`.

Δt = 3minutes
stop_time = 4days

simulation = Simulation(model, Δt = Δt, stop_time = stop_time)

# We add a callback to print a message about how the simulation is going,

using Printf

wall_clock = Ref(time_ns())

function print_progress(sim)

    elapsed = 1e-9 * (time_ns() - wall_clock[])

    msg = @sprintf("iteration: %d, time: %s, wall time: %s, max|w|: %6.3e, m s⁻¹\n",
                   iteration(sim), prettytime(sim), prettytime(elapsed),
                   maximum(abs, sim.model.velocities.w))

    wall_clock[] = time_ns()

    @info msg

    return nothing
end

simulation.callbacks[:print_progress] = Callback(print_progress, IterationInterval(200))

# ## Diagnostics/Output

# Add some diagnostics.

b = model.tracers.b
u, v, w = model.velocities

U = Field(Average(u))

u′ = u - U

N² = ∂z(b)

filename = "barotropic_tide"
save_fields_interval = 30minutes

simulation.output_writers[:fields] = JLD2OutputWriter(model, (; u, u′, w, b, N²);
                                                      filename,
                                                      schedule = TimeInterval(save_fields_interval),
                                                      overwrite_existing = true)

# We are ready -- let's run!

@info "Running the simulation..."

run!(simulation)

@info "Simulation completed in " * prettytime(simulation.run_wall_time)

# ## Visualization

# Now let's visualize our resutls! We use `CairoMakie` in this example.
# On a system with OpenGL `using GLMakie` is more convenient as figures will be
# displayed on the screen.

using CairoMakie

# We load the saved buoyancy output on the top, bottom, and east surface as `FieldTimeSeries`es.

saved_output_filename = filename * ".jld2"

u′_t = FieldTimeSeries(saved_output_filename, "u′")
 w_t = FieldTimeSeries(saved_output_filename, "w")
N²_t = FieldTimeSeries(saved_output_filename, "N²")

times = u′_t.times
nothing #hide

# We build the coordinates. We rescale horizontal coordinates so that they correspond to kilometers.

xu,  yu,  zu  = nodes(u′_t[1])
xw,  yw,  zw  = nodes(w_t[1])
xN², yN², zN² = nodes(N²_t[1])
nothing #hide

# A utility to mask the region that is within the immersed boundary with `NaN`s.

using Oceananigans.ImmersedBoundaries: mask_immersed_field!

function mask_and_get_interior(φ_t, n; value=NaN)
    mask_immersed_field!(φ_t[n], value)
    return interior(φ_t[n], :, 1, :)
end

# We use Makie's `Observable` to animate the data. To dive into how `Observable`s work we
# refer to [Makie.jl's Documentation](https://makie.juliaplots.org/stable/documentation/nodes/index.html).

n = Observable(1)

title = @lift @sprintf("t = %1.2f days = %1.2f T₂",
                       round(times[$n] / day, digits=2) , round(times[$n] / T₂, digits=2))

u′ₙ = @lift mask_and_get_interior(u′_t, $n)
 wₙ = @lift mask_and_get_interior(w_t, $n)
N²ₙ = @lift mask_and_get_interior(N²_t, $n)

axis_kwargs = (xlabel = "x [km]",
               ylabel = "z [m]",
               limits = ((-Lx / 2e3, Lx / 2e3), (-H, 0)),
               titlesize = 20)

ulim   = 0.8 * maximum(abs, u′_t[end])
wlim   = 0.8 * maximum(abs, w_t[end])

fig = Figure(resolution = (700, 900))

ax_u = Axis(fig[2, 1];
            title = "u′-velocity", axis_kwargs...)

ax_w = Axis(fig[3, 1];
            title = "w-velocity", axis_kwargs...)

ax_N² = Axis(fig[4, 1];
             title = "stratification", axis_kwargs...)

fig[1, :] = Label(fig, title, fontsize=24, tellwidth=false)

hm_u = heatmap!(ax_u, xu/1e3, zu, u′ₙ;
                colorrange = (-ulim, ulim),
                colormap = :balance)
Colorbar(fig[2, 2], hm_u, label = "m s⁻¹")

hm_w = heatmap!(ax_w, xw/1e3, zw, wₙ;
                colorrange = (-wlim, wlim),
                colormap = :balance)
Colorbar(fig[3, 2], hm_w, label = "m s⁻¹")

hm_N² = heatmap!(ax_N², xN²/1e3, zN², N²ₙ;
                 colorrange = (0.95Nᵢ², 1.05Nᵢ²),
                 colormap = :thermal)
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

# ![](barotropic_tide.mp4)
