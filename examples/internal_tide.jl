# # Internal tide with open boundaries and z-star
#
# In this example, we show how internal tides generated over a seamount radiate
# out through open boundaries, using a z-star (free-surface-following) vertical
# coordinate.
#
# This builds on the periodic `internal_tide.jl` example, but replaces the
# periodic x-topology with `Bounded` and `Radiation` open boundary conditions
# that allow waves to exit the domain cleanly.
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
using Oceananigans.Grids: znode, xnode
using Oceananigans.BoundaryConditions: FlatherBoundaryCondition

# ## Grid
#
# We create a two-dimensional `RectilinearGrid` that is `Bounded` in ``x`` and ``z``.
# The ``z`` coordinate uses `MutableVerticalDiscretization` to enable the z-star
# (free-surface-following) vertical coordinate.

Nx, Nz = 256, 128
H, L = 2kilometers, 1000kilometers

z = MutableVerticalDiscretization((-H, 0))

underlying_grid = RectilinearGrid(size = (Nx, Nz), halo = (5, 5),
                                  x = (-L, L), z = (-H, 0),
                                  topology = (Bounded, Flat, Bounded))

# Now we create a Gaussian seamount at the center of the domain:
#
# ```math
# h(x) = -H + h_0 \exp(-x^2 / 2\sigma^2)
# ```

h₀ = 250meters
width = 20kilometers
hill(x) = h₀ * exp(-x^2 / 2width^2)
bottom(x) = - H + hill(x)

grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(bottom))

# ## Tidal forcing
#
# We force a barotropic tide by imposing an oscillating cross-shelf flow at the open
# boundaries. We use the lunar semi-diurnal ``M_2`` tide, which oscillates as
# ``\sin(\omega_2 t)`` with frequency ``\omega_2 = 2 \pi / T_2`` and period
# ``T_2 = 12.421 \, \mathrm{hours}``.
#
# The Coriolis frequency enters the tidal response, so we first construct a Coriolis on an
# ``f``-plane at mid-latitudes.

coriolis = FPlane(latitude = -45)

T₂ = 12.421hours
ω₂ = 2π / T₂ # radians/sec

# The strength of the tide is set by the excursion parameter ``\epsilon``, the nondimensional
# ratio of the horizontal distance a fluid parcel travels over a tidal period to the width
# ``\sigma`` of the hill,
#
# ```math
# \epsilon = \frac{U_2 / \omega_2}{\sigma} ,
# ```
#
# so prescribing ``\epsilon`` fixes the tidal velocity amplitude ``U_2 = \epsilon \, \omega_2 \, \sigma``.

ϵ = 0.1 # excursion parameter
U₂ = ϵ * ω₂ * width

# A Fourier decomposition of the inviscid, linearized momentum equations relates the tidal
# velocity to the amplitude ``A_2`` of the barotropic forcing that sustains it,
#
# ```math
# U_2 = \frac{\omega_2}{\omega_2^2 - f^2} A_2 ,
# ```
#
# which we invert for ``A_2``.

A₂ = U₂ * (ω₂^2 - coriolis.f^2) / ω₂

# ## Boundary conditions
#
# We use `NormalFlow` boundary conditions with the `Radiation` scheme (Orlanski 1976,
# Marchesiello et al. 2001) on ``u`` at the east and west boundaries. The
# `Radiation` scheme diagnoses the phase speed of outgoing waves from interior
# gradients and advects boundary values accordingly. Adaptive nudging
# provides weak relaxation on outflow and stronger relaxation on inflow.
#
# For the split-explicit barotropic solver, we impose a `Flather` characteristic
# condition on the barotropic transport, ensuring proper barotropic wave radiation
# at the free surface level.

Nᵢ² = 1e-4  # [s⁻²] initial buoyancy frequency / stratification

@inline tidal_forcing(z, t, p) = p.U₂ * sin(p.ω₂ * t)

u_east_bc = NormalFlowBoundaryCondition(tidal_forcing; scheme = Radiation(), parameters = (; U₂, ω₂))
u_west_bc = NormalFlowBoundaryCondition(tidal_forcing; scheme = Radiation(), parameters = (; U₂, ω₂))
u_bcs     = FieldBoundaryConditions(east = u_east_bc, west = u_west_bc)

@inline barotropic_tidal_forcing(i, j, grid, clock, model_fields) = (U₂ * sin(ω₂ * clock.time) * 2kilometers, 0)

U_west_bc = FlatherBoundaryCondition(barotropic_tidal_forcing; discrete_form = true)
U_east_bc = FlatherBoundaryCondition(barotropic_tidal_forcing; discrete_form = true)
U_bcs     = FieldBoundaryConditions(grid, (Face(), Center(), nothing); east = U_east_bc, west = U_west_bc)

# Internal waves carry buoyancy perturbations along with velocity, so the
# buoyancy tracer also needs open boundary conditions. Without them, the
# default `NoFlux` (zero-gradient) condition reflects buoyancy signals at
# the boundary, which drives reflected velocity waves. We nudge toward the
# background stratification `N²z` on inflow.

@inline b_background(z, t, Nᵢ²) = Nᵢ² * z
b_east_bc = ValueBoundaryCondition(b_background; scheme = Radiation(), parameters = Nᵢ²)
b_west_bc = ValueBoundaryCondition(b_background; scheme = Radiation(), parameters = Nᵢ²)
b_bcs     = FieldBoundaryConditions(east = b_east_bc, west = b_west_bc)

# ## Sponge layers
#
# Radiation open boundary conditions are imperfect: Orlanski's scheme diagnoses a single phase
# speed, while internal waves carry energy in several baroclinic modes that travel at different
# speeds. A sponge layer near each boundary damps the waves before they reach it, reducing
# reflections. The mask uses a ``\cos^2(\pi d / 2 W)`` profile that ramps smoothly from ``0`` a
# distance ``W`` into the interior to ``1`` at the boundary, where ``d`` is the distance to the
# nearest boundary.
#
# We use `discrete_form = true` so the forcing functions index the fields and grid coordinates
# directly, which sidesteps coordinate-mapping issues with the `Flat` ``y``-topology.

@inline function sponge_mask(x, p)
    d_west = x + p.Lx
    d_east = p.Lx - x
    d = min(d_west, d_east)
    return ifelse(d < p.W, cospi(d / (2 * p.W))^2, zero(d))
end

# The momentum sponge relaxes ``u`` toward the imposed barotropic tide `ut`, leaving the tide
# intact while damping the baroclinic perturbation riding on top of it.

@inline function u_sponge_forcing(i, j, k, grid, clock, fields, p)
    x = xnode(i, grid, Face())
    mask = sponge_mask(x, p)
    ut = tidal_forcing(1, clock.time, p)
    return -p.rate * mask * (@inbounds fields.u[i, j, k] - ut)
end

# The buoyancy sponge relaxes ``b`` toward the background stratification ``N^2 z``.

@inline b_bg(z, p) = p.N² * z

@inline function b_sponge_forcing(i, j, k, grid, clock, fields, p)
    x = xnode(i, grid, Center())
    z = znode(k, grid, Center())
    mask = sponge_mask(x, p)
    bt = b_bg(z, p)
    return -p.rate * mask * (@inbounds fields.b[i, j, k] - bt)
end

sponge_params = (; Lx = L, W = 200kilometers, rate = 1 / 30minutes, U₂, ω₂, N² = Nᵢ²)

u_sponge = Forcing(u_sponge_forcing, discrete_form = true, parameters = sponge_params)
b_sponge = Forcing(b_sponge_forcing, discrete_form = true, parameters = sponge_params)


# ## Model
#
# We build a `HydrostaticFreeSurfaceModel` with `SplitExplicitFreeSurface`.
# Since the barotropic transport has prescribed normal-flow boundaries (Flather),
# the solver automatically re-applies the boundary conditions during the
# barotropic substep loop.

free_surface = SplitExplicitFreeSurface(grid; substeps = 30)

model = HydrostaticFreeSurfaceModel(grid; 
                                    coriolis,
                                    buoyancy = BuoyancyTracer(),
                                    tracers = :b,
                                    momentum_advection = WENO(),
                                    tracer_advection = WENO(),
                                    free_surface,
                                    timestepper = :SplitRungeKutta3,
                                    forcing = (; u = u_sponge, b = b_sponge),
                                    boundary_conditions = (; u = u_bcs, U = U_bcs, b = b_bcs))

# Initialize with the tidal flow and a linear stratification.
bᵢ(x, z) = Nᵢ² * z
gaussian(x, σ) = exp(-x^2 / 2σ^2)
ηᵢ(x, z) = gaussian(x, 100kilometers) # sin(π * x / L) # 
set!(model, b=bᵢ) #, η=ηᵢ)

# ## Simulation

Δt = 10minutes
stop_time = 10days
simulation = Simulation(model; Δt, stop_time)

# We add a callback to print a message about how the simulation is going.

using Printf

wall_clock = Ref(time_ns())

function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])

    msg = @sprintf("Iter: %d, time: %s, wall time: %s, max|w|: %6.3e, m s⁻¹\n",
                   iteration(sim), prettytime(sim), prettytime(elapsed),
                   maximum(abs, sim.model.velocities.w))

    wall_clock[] = time_ns()

    @info msg

    return nothing
end

add_callback!(simulation, progress, name=:progress, IterationInterval(200))
nothing #hide

# ## Diagnostics/Output
#
# Rather than ``u`` itself we save its deviation from the instantaneous domain average,
# ``u' = u - (L_x H)^{-1} \int u \, \mathrm{d}x \, \mathrm{d}z``, which subtracts the barotropic
# tide and exposes the baroclinic internal-tide response. We also save the vertical velocity
# ``w``, the buoyancy ``b``, the stratification ``N^2 = \partial_z b``, and the free surface ``\eta``.

b = model.tracers.b
u, v, w = model.velocities
U = Field(Average(u))
u′ = u - U
N² = ∂z(b)
η = model.free_surface.displacement
Ub = model.free_surface.barotropic_velocities.U

filename = "internal_tide_open_boundaries"
save_fields_interval = 30minutes

simulation.output_writers[:fields] = JLD2Writer(model, (; u, u′, w, b, N², η, Ub); filename,
                                                schedule = TimeInterval(10minutes),
                                                overwrite_existing = true)

# We are ready -- let's run!

run!(simulation)

# ## Load output

saved_output_filename = filename * ".jld2"

 u_t = FieldTimeSeries(saved_output_filename, "u")
u′_t = FieldTimeSeries(saved_output_filename, "u′")
 w_t = FieldTimeSeries(saved_output_filename, "w")
N²_t = FieldTimeSeries(saved_output_filename, "N²")
 η_t = FieldTimeSeries(saved_output_filename, "η")
 U_t = FieldTimeSeries(saved_output_filename, "Ub")

umax = maximum(abs, u′_t)
wmax = maximum(abs, w_t)

times = u′_t.times
nothing #hide

# ## Visualize
#
# We visualize the baroclinic velocity perturbation, vertical velocity,
# stratification, and free surface displacement. With open boundaries, the
# internal waves should radiate cleanly out of the domain without reflections.

using CairoMakie

n = Observable(1)

title = @lift @sprintf("t = %1.2f days = %1.2f T₂",
                       round(times[$n] / day, digits=2) , round(times[$n] / T₂, digits=2))

u′ₙ = @lift interior(u′_t[$n], :, 1, :)
 wₙ = @lift interior( w_t[$n], :, 1, :)
N²ₙ = @lift interior(N²_t[$n], :, 1, :)
 ηₙ = @lift interior( η_t[$n], :, 1, :)
 Uₙ = @lift interior( U_t[$n], :, 1, :)
u1ₙ = @lift interior( u_t[$n], 1, 1, :)
ueₙ = @lift interior( u_t[$n], grid.Nx+1, 1, :)

axis_kwargs = (xlabel = "x [km]",
               ylabel = "z [m]",
               titlesize = 20)

fig = Figure(size = (700, 1100))

fig[1, :] = Label(fig, title, fontsize=24, tellwidth=false)

ax_u = Axis(fig[2, 1:2]; title = "u'-velocity", axis_kwargs...)
hm_u = heatmap!(ax_u, u′ₙ; nan_color=:gray, colorrange=(-umax, umax), colormap=:balance)
Colorbar(fig[2, 3], hm_u, label = "m s⁻¹")

ax_w = Axis(fig[3, 1:2]; title = "w-velocity", axis_kwargs...)
hm_w = heatmap!(ax_w, wₙ; nan_color=:gray, colorrange=(-wmax, wmax), colormap=:balance)
Colorbar(fig[3, 3], hm_w, label = "m s⁻¹")

ax_N² = Axis(fig[4, 1:2]; title = "stratification N²", axis_kwargs...)
hm_N² = heatmap!(ax_N², N²ₙ; nan_color=:gray, colorrange=(0.9Nᵢ², 1.1Nᵢ²), colormap=:magma)
Colorbar(fig[4, 3], hm_N², label = "s⁻²")

ax_η = Axis(fig[5, 1]; title = "free surface η",
            xlabel = "x [km]",
            titlesize = 20,
            ylabel = "η [m]")
ηₙ_line = @lift interior(η_t[$n], :, 1, 1)
lines!(ax_η, ηₙ_line, color=:dodgerblue)
ylims!(ax_η, -4.0, 4.0)

ax_U = Axis(fig[5, 2]; title = "Boundary velocities",
            xlabel = "x [km]",
            titlesize = 20,
            ylabel = "ub [m]")
lines!(ax_U, u1ₙ, color=:dodgerblue)
lines!(ax_U, ueₙ, color=:blue)

fig

# Finally, we can record a movie.

@info "Making an animation from saved data..."

frames = 1:length(times)

record(fig, filename * ".mp4", frames, framerate=16) do i
    @info string("Plotting frame ", i, " of ", frames[end])
    n[] = i
end
nothing #hide

# ![](internal_tide_open_boundaries.mp4)
