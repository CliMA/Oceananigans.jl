# # [Hydrostatic lock exchange with CATKEVerticalDiffusivity](@id hydrostatic_lock_exchange_example)
#
# This example simulates a lock exchange problem on a slope. It demonstrates:
#
#  * How to set up a 2D grid with a sloping bottom using an immersed boundary
#  * Initializing a hydrostatic free surface model
#  * Including variable density initial conditions
#  * Applying bottom drag boundary conditions on immersed boundaries using [`BulkDrag`](@ref)
#  * Saving outputs of the simulation
#  * Creating an animation with CairoMakie
#
# ### The Lock Exchange Problem
#
# In a lock exchange, two fluids of different densities (due to temperature, salinity, etc.)
# are separated by a "lock" at time ``t = 0``. Once the lock is removed, the fluids
# interact and form gravity currents.
# The lock exchange represents scenarios where waters of different salinities or temperatures
# meet and form sharp density gradients, for example in estuaries or in the Denmark Strait overflow.
# Its evolution can be described by the hydrostatic Boussinesq equations. In this example, we use buoyancy
# as a tracer; see the [Boussinesq approximation section](@ref boussinesq_approximation) for
# more details on the Boussinesq approximation.

# ## Install dependencies
#
# First let's make sure we have all required packages installed.

# ```julia
# using Pkg
# pkg"add Oceananigans, CairoMakie"
# ```

# ## Import Required Packages

using Oceananigans
using Oceananigans.Units
using Printf
using CairoMakie

# ## Set up a 2D rectilinear grid

Nx = 128
Nz = 64
L = 8kilometers   # horizontal length
H = 50meters      # depth

# We wrap `z` into a `MutableVerticalDiscretization`. This allows the `HydrostaticFreeSurfaceModel`
# (which we construct further down) to use a time-evolving, free-surface-following
# `ZStarCoordinate` vertical coordinate.

x = (-L/8, 7L/8)
z = MutableVerticalDiscretization((-H, 0))

underlying_grid = RectilinearGrid(; size=(Nx, Nz), x, z, halo=(5, 5),
                                    topology=(Bounded, Flat, Bounded))

# The bottom slopes upward from ``z = -H`` at ``x = 0`` to ``z = -H/2`` at ``x = L``.
# We describe the sloping bottom with a partial-cell immersed boundary.

slope = H / 2L
bottom(x) = -H + slope * x

grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(bottom))

# ## Set up a bottom drag boundary condition on the immersed boundary
#
# To apply drag to the flow, we use `BulkDrag`, which implements
# quadratic drag proportional to ``Cᴰ |U| u``, where ``Cᴰ`` is the drag coefficient and
# ``|U| = \sqrt{u² + v²}`` is the horizontal speed. We use a drag coefficient ``Cᴰ = 0.002``,
# a reasonable value for seafloor drag.

Cᴰ = 0.002
drag = BulkDrag(coefficient=Cᴰ)

# `BulkDrag` can be applied both to domain boundaries (like the bottom of a `Bounded` grid)
# and to immersed boundaries. Here we apply it to the immersed sloping bottom boundary:

u_bcs = FieldBoundaryConditions(bottom=drag, immersed=drag)

# In a 2D simulation with `Flat` in the ``y``-direction, we don't need boundary conditions for ``v``.

# ## Initialize the model
#
#  * We use a hydrostatic model because the horizontal scale of the flow is much larger than its vertical scale
#  * Buoyancy ``b`` is a tracer that determines the fluid density
#  * The vertical closure [`CATKEVerticalDiffusivity`](@ref) parameterizes small-scale vertical turbulence
#  * Weighted Essentially Non-Oscillatory (WENO) advection schemes capture sharp changes in density

model = HydrostaticFreeSurfaceModel(grid;
                                    tracers = :b,
                                    buoyancy = BuoyancyTracer(),
                                    closure = CATKEVerticalDiffusivity(),
                                    momentum_advection = WENO(order=5),
                                    tracer_advection = WENO(order=7),
                                    boundary_conditions = (; u=u_bcs),
                                    free_surface = SplitExplicitFreeSurface(grid; substeps=20))

# ## Set variable density initial conditions
#
# The lock at ``x = L/2`` separates light fluid on the left from denser fluid on the right.

bᵢ(x, z) = x > L/2 ? 0.01 : 0.06
set!(model, b=bᵢ)

# ## Construct a Simulation
#
# Fast wave speeds make the equations stiff, so the CFL condition restricts the time step to
# small values to maintain numerical stability.

simulation = Simulation(model, Δt=1second, stop_time=6hours)

# The [`TimeStepWizard`](@ref) is incorporated in the simulation via the
# [`conjure_time_step_wizard!`](@ref) helper function and it ensures stable
# time-stepping with a Courant–Friedrichs–Lewy (CFL) number of 0.3.

conjure_time_step_wizard!(simulation, cfl=0.3)

# ## Track simulation progress
#
# We add a callback that prints the simulation progress alongside some flow statistics.

function progress(sim)
    @info @sprintf("Iter: %6d, time: %s, Δt: %s, wall time: %s, max|w| = %6.3e m s⁻¹",
                   iteration(sim),
                   prettytime(sim),
                   prettytime(sim.Δt),
                   prettytime(sim.run_wall_time),
                   maximum(abs, sim.model.velocities.w))
    return nothing
end

add_callback!(simulation, progress, IterationInterval(1000))

# ## Add output writer
#
# Here, we construct a `JLD2Writer` to save buoyancy ``b``, turbulent kinetic energy ``e``,
# horizontal velocity ``u``, and stratification ``N² = ∂b/∂z`` every 2 minutes.

b = model.tracers.b
e = model.tracers.e
u = model.velocities.u
N² = ∂z(b)

filename = "hydrostatic_lock_exchange.jld2"
simulation.output_writers[:fields] = JLD2Writer(model, (; b, e, u, N²);
                                                filename,
                                                schedule = TimeInterval(2minutes),
                                                overwrite_files = true)

# ## Run the simulation

## Fail the docs build if this simulation produces NaNs #hide
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation) #hide
run!(simulation)

@info "Simulation finished. Output saved to $(filename)"

# ## Load the saved time series

ut = FieldTimeSeries(filename, "u")
N²t = FieldTimeSeries(filename, "N²")
bt = FieldTimeSeries(filename, "b")
et = FieldTimeSeries(filename, "e")
times = bt.times

# ## Visualize the simulation output

# We use Makie's `Observable` to animate the data. To dive into how `Observable`s work we
# refer to [Makie.jl's Documentation](https://docs.makie.org/stable/explanations/observables).

n = Observable(1)

title = @lift "t = " * prettytime(times[$n])

uₙ = @lift ut[$n]
N²ₙ = @lift N²t[$n]
bₙ = @lift bt[$n]
eₙ = @lift et[$n]
nothing #hide

# We use the last snapshot to set the color ranges.

umax = maximum(abs, ut[end])
N²max = maximum(abs, N²t[end])
bmax = maximum(abs, bt[end])
emax = maximum(abs, et[end])
nothing #hide

# We visualize ``b``, ``e``, ``u``, and ``N²``.

nan_color = :grey
axis_kwargs = (xlabel = "x (m)", ylabel = "z (m)",
               limits = (x, (-H, 0)), titlesize = 18)

fig = Figure(size = (800, 900))
fig[1, :] = Label(fig, title, fontsize = 24, tellwidth = false)

ax_b = Axis(fig[2, 1]; title = "b (buoyancy)", axis_kwargs...)
hm_b = heatmap!(ax_b, bₙ; nan_color, colorrange = (0, bmax), colormap = :thermal)
Colorbar(fig[2, 2], hm_b, label = "m s⁻²")

ax_e = Axis(fig[3, 1]; title = "e (turbulent kinetic energy)", axis_kwargs...)
hm_e = heatmap!(ax_e, eₙ; nan_color, colorrange = (0, emax), colormap = :magma)
Colorbar(fig[3, 2], hm_e, label = "m² s⁻²")

ax_u = Axis(fig[4, 1]; title = "u (horizontal velocity)", axis_kwargs...)
hm_u = heatmap!(ax_u, uₙ; nan_color, colorrange = (-umax, umax), colormap = :balance)
Colorbar(fig[4, 2], hm_u, label = "m s⁻¹")

ax_N² = Axis(fig[5, 1]; title = "N² (stratification)", axis_kwargs...)
hm_N² = heatmap!(ax_N², N²ₙ; nan_color, colorrange = (-N²max/4, N²max), colormap = :haline)
Colorbar(fig[5, 2], hm_N², label = "s⁻²")

record(fig, "hydrostatic_lock_exchange.mp4", 1:length(times); framerate = 8) do i
    n[] = i
end
nothing #hide

# The visualization shows the time evolution of buoyancy ``b``, turbulent kinetic energy ``e``,
# horizontal velocity ``u``, and stratification ``N²``.
# Initially, the two water masses are separated horizontally by a sharp density interface.
# As the flow evolves, gravity currents form and the dense fluid moves beneath the lighter fluid.
# This shows the characteristic transition from horizontal to vertical density separation
# for the lock exchange problem.

# ![](hydrostatic_lock_exchange.mp4)
