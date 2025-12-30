# # [Hydrostatic lock exchange with CATKEVerticalDiffusivity](@id hydrostatic_lock_exchange_example)
#
# This example simulates a lock exchange problem on a slope. It demonstrates:
#
#  * How to set up a 2D grid with a sloping bottom with an immersed boundary
#  * Initializing a hydrostatic free surface model
#  * Including variable density initial conditions
#  * Applying bottom drag boundary conditions on immersed boundaries using [`BulkDrag`](@ref)
#  * Saving outputs of the simulation
#  * Creating an animation with CairoMakie
#
# ### The Lock Exchange Problem
#
# This use case is a basic example where there are fluids of two different densities (due to
# temperature, salinity, etc.) that are separated by a ‘lock’ at time ``t=0``, and we
# calculate the evolution of how these fluids interact as time progresses.
# This lock exchange implementation can be a representation of scenarios where water of
# different salinities or temperatures meet and form sharp density gradients.
# For example, in estuaries or in the Denmark Strait overflow.
# Solutions of this problem describe how the fluids interact with each other as time evolves
# and can be described by hydrostatic Boussinesq equations. In this example, we use buoyancy
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

# ## Set up 2D Rectilinear Grid

# Set resolution of the simulation grid
Nx = 128
Nz = 64
L = 8kilometers   # horizontal length
H = 50meters      # depth

# Set domain:
# We wrap `z` into a MutableVerticalDiscretization. This allows the HydrostaticFreeSurface
# model (which we construct further down) to use a time-evolving `ZStarCoordinate`
# free-surface–following vertical coordinate.
x = (-L/8, 7L/8)
z  = MutableVerticalDiscretization((-50, 0))

# Initialize the grid:
# Additional details on the grid set up can be found in Grids section of the documentation

underlying_grid = RectilinearGrid(; size=(Nx, Nz), x, z, halo=(5, 5),
                                    topology=(Bounded, Flat, Bounded))


# Add a slope at the bottom of the grid
h_left = -H
h_right = -H/2
slope = (h_right - h_left) / L
bottom(x) = h_left + slope * x

# Use an immersed boundary with grid fitted bottom to describe the sloped bottom of the domain
grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(bottom))


# ## Set up a bottom drag boundary condition on the immersed boundary
#
# To apply drag to the flow, we use `BulkDrag`, which implements
# quadratic drag proportional to `Cᴰ |U| u`, where `Cᴰ` is the drag coefficient and
# `|U| = sqrt(u² + v²)` is the horizontal speed. We use a drag coefficient `Cᴰ = 0.002`,
# a reasonable value for seafloor drag.

Cᴰ = 0.002
drag = BulkDrag(coefficient=Cᴰ)

# `BulkDrag` can be applied both to domain boundaries (like the bottom of a `Bounded` grid)
# and to immersed boundaries. Here we apply it to the immersed sloping bottom boundary:

u_bcs = FieldBoundaryConditions(bottom=drag, immersed=drag)

# Note: in a 2D simulation with `Flat` in the y-direction, we don't need v boundary conditions.

# ## Initialize the model
#
#  * Want to use a hydrostatic model since horizontal motion may be more significant than vertical motion
#  * Tracers act as markers within the fluid to track movement and dispersion
#  * Vertical closure [`CATKEVerticalDiffusivity`](@ref) handles small-scale vertical turbulence
#  * Weighted Essentially Non-Oscillatory (WENO) methods are useful for capturing sharp changes in density

model = HydrostaticFreeSurfaceModel(; grid,
                                    tracers = :b,
                                    buoyancy = BuoyancyTracer(),
                                    closure = CATKEVerticalDiffusivity(),
                                    momentum_advection = WENO(order=5),
                                    tracer_advection = WENO(order=7),
                                    boundary_conditions = (; u=u_bcs),
                                    free_surface = SplitExplicitFreeSurface(grid; substeps=20))

# ## Set variable density initial conditions

# Set initial conditions for lock exchange with different buoyancies.
bᵢ(x, z) = x > L/2 ? 0.01 : 0.06
set!(model, b=bᵢ)

# ## Construct a Simulation
#
# Fast wave speeds make the equations stiff, so the CFL condition restricts the timestep to
# adequately small values to maintain numerical stability.

# Set the timesteps
Δt = 1second
stop_time = 6hours
simulation = Simulation(model; Δt, stop_time)

# The [`TimeStepWizard`](@ref) is incorporated in the simulation via the
# [`conjure_time_step_wizard!`](@ref) helper function and it ensures stable
# time-stepping with a Courant-Freidrichs-Lewy (CFL) number of 0.35.
conjure_time_step_wizard!(simulation, cfl=0.3)

# ## Track Simulation Progress

## Wall clock represents the real world time as opposed to simulation time
wall_clock = Ref(time_ns())

## Define callback function to show how the simulation is progressing alongside with some flow statistics.
function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])
    msg = @sprintf("Iter: %6d, time: %s, Δt: %s, wall: %s, max|w| = %6.3e m s⁻¹",
                   iteration(sim),
                   prettytime(sim),
                   prettytime(sim.Δt),
                   prettytime(elapsed),
                   maximum(abs, sim.model.velocities.w))
    @info msg
    wall_clock[] = time_ns()
    return nothing
end

add_callback!(simulation, progress, IterationInterval(1000))

# ## Add output writer
#
# Here, we construct a JLD2Writer to save some output every `save_interval`.

b = model.tracers.b
e = model.tracers.e
u, v, w = model.velocities
N² = ∂z(b)

filename = "hydrostatic_lock_exchange.jld2"
save_interval = 2minutes
simulation.output_writers[:fields] = JLD2Writer(model, (; b, e, u, N²);
                                                filename = filename,
                                                schedule = TimeInterval(save_interval),
                                                overwrite_existing = true)

# ## Run Simulation

run!(simulation)

@info "Simulation finished. Output saved to $(filename)"

# ## Load Saved TimeSeries Values

ut = FieldTimeSeries(filename, "u")
N²t = FieldTimeSeries(filename, "N²")
bt = FieldTimeSeries(filename, "b")
et = FieldTimeSeries(filename, "e")
times = bt.times

# ## Visualize Simulation Outputs

# We use Makie's `Observable` to animate the data. To dive into how `Observable`s work we
# refer to [Makie.jl's Documentation](https://docs.makie.org/stable/explanations/observables).

n = Observable(1)

title = @lift @sprintf("t = %5.2f hours", times[$n] / hour)

un = @lift ut[$n]
N²n = @lift N²t[$n]
bn = @lift bt[$n]
en = @lift et[$n]
nothing #hide

# For visualization color ranges (use last snapshot)
umax = maximum(abs, ut[end])
N²max = maximum(abs, N²t[end])
bmax = maximum(abs, bt[end])
emax = maximum(abs, et[end])
nothing #hide

# Use snapshots to create Makie visualization for ``b``,  ``e``, ``u``, and ``N^2``.

nan_color = :grey
axis_kwargs = (xlabel = "x [m]", ylabel = "z [m]",
               limits = ((0, L), (-H, 0)), titlesize = 18)

fig = Figure(size = (800, 900))
fig[1, :] = Label(fig, title, fontsize = 24, tellwidth = false)

ax_b = Axis(fig[2, 1]; title = "b (buoyancy)", axis_kwargs...)
hm_b = heatmap!(ax_b, bn; nan_color, colorrange = (0, bmax), colormap = :thermal)
Colorbar(fig[2, 2], hm_b, label = "m s⁻²")

ax_e = Axis(fig[3, 1]; title = "e (turbulent kinetic energy)", axis_kwargs...)
hm_e = heatmap!(ax_e, en; nan_color, colorrange = (0, emax), colormap = :magma)
Colorbar(fig[3, 2], hm_e, label = "m² s⁻²")

ax_u = Axis(fig[4, 1]; title = "u (horizontal velocity)", axis_kwargs...)
hm_u = heatmap!(ax_u, un; nan_color, colorrange = (-umax, umax), colormap = :balance)
Colorbar(fig[4, 2], hm_u, label = "m s⁻¹")

ax_N² = Axis(fig[5, 1]; title = "N² (stratification)", axis_kwargs...)
hm_N² = heatmap!(ax_N², N²n; nan_color, colorrange = (-N²max/4, N²max), colormap = :haline)
Colorbar(fig[5, 2], hm_N², label = "s⁻²")

Nt = length(times)
record(fig, "hydrostatic_lock_exchange.mp4", 1:Nt; framerate = 8) do nn
    @info "Animating frame $nn out of $Nt"
    n[] = nn
end

# The visualization shows the time evolution of buoyancy ``b``, turbulent kinetic energy ``e``,
# stratification ``N²``, and zonal velocity ``u``.
# Initially, the two water masses are separated horizontally by a sharp density interface.
# As the flow evolves, gravity currents form and the dense fluid moves beneath the lighter fluid.
# This shows the characteristic transition from horizontal to vertical density separation
# for the lock exchange problem.

# ![](hydrostatic_lock_exchange.mp4)
