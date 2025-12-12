# # [Lock Exchange Example](@id lock_exchange_example)
#
# This is a simple example of a lock exchange problem on a slope. This example demonstrates: 
#
#  * How to set up a 2D grid with a sloping bottom with an immersed boundary
#  * Initializing a hydrostatic free surface model 
#  * Including variable density initial conditions 
#  * Saving outputs of the simulation 
#  * Creating an animation with CairoMakie
# 
# ### The Lock Exchange Problem 
#     This use case is a basic example where there are fluids of two different densities (due to temperature, salinity, etc.) 
# that are separated by a ‘lock’ at time t=0, and we calculate the evolution of how these fluids interact as time progresses. 
# This lock exchange implementation can be a representation of 
# scenarios where water of different salinities or temperatures meet and form sharp density gradients. 
# For example, in estuaries or in the Denmark Strait overflow.
# Solutions of this problem describe how the fluids interact with each other as time evolves and can be described by 
# hydrostatic Boussinesq equations: 
# ```math
# \frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \textbf{v}) = 0
# ```
# where `ρ` represents density. In the case of this model, we use boyancy as a tracer, which relates to the density: 
# ```math
# b = -g \frac{\rho - \rho_{0}}{\rho_{0}}
# ```
# 
# For a detailed explanation of the Boussinesq formulation for buoyant flows,
# see Barletta (2022), *The Boussinesq approximation for buoyant flows*,
# *Mechanics Research Communications* 124, 103939.


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
using Oceananigans.Grids
using Oceananigans.Diagnostics
using Oceananigans.OutputWriters
using Oceananigans.Operators

using Printf
using CairoMakie


# ## Set up 2D Rectilinear Grid

# Set resolution of the simulation grid 
Nx, Nz = 128, 64

# Set grid size 
L = 8kilometers   # horizontal length
H = 50meters      # depth  

# Set domain 
# MutableVerticalDiscretization defines a time-evolving vertical coordinate,
# for example free-surface–following or terrain-following (sigma) coordinates.
x = (0, L)
z  = MutableVerticalDiscretization((-50, 0))

# Initialize the grid: 
# Additional details on the grid set up can be found in Grids section of the documentation 

underlying_grid = RectilinearGrid(size=(Nx, Nz); x, z, halo=(5, 5), topology=(Bounded, Flat, Bounded))


# Add a slope at the bottom of the grid 
h_left = -H
h_right = -25meters
slope = (h_right - h_left) / L
bottom(x) = h_left + slope * x

# Use an immersed boundary with grid fitted bottom to describe the sloped bottom of the domain 
grid = ImmersedBoundaryGrid(underlying_grid, PartialCellBottom(bottom))


# ## Initialize the Model 
#
#  * Want to use a hydrostatic model since horizontal motion may be more significant than verticle motion 
#  * Tracers act as markers within the fluid to track movement and dispersion
#  * Vertical closure CATKEVerticalDiffusivity handles small-scale vertical turbulence 
#  * Weighted Essentially Non-Oscillatory (WENO) methods are useful for capturing sharp changes in density
#  * ZStarCoordinate method allows for the top of the grid to move with the free surface

model = HydrostaticFreeSurfaceModel(; grid,
    tracers = (:b, :e),      
    buoyancy = BuoyancyTracer(),
    closure = CATKEVerticalDiffusivity(),
    momentum_advection = WENO(order=5), 
    tracer_advection = WENO(order=7), 
    vertical_coordinate = ZStarCoordinate(grid), 
    free_surface = SplitExplicitFreeSurface(grid; substeps=10)
)


# ## Set Variable Density Initial Conditions 

# Set initial conditions for lock exchange with different boyancies  
bᵢ(x, z) = x > 4kilometers ? 0.01 : 0.06
set!(model, b=bᵢ)


# ## Defining Simulation Timestep 
# 
# Fast wave speeds make the equations stiff, so the CFL condition forces a small timestep Δt
# when using explicit time-stepping to maintain numerical stability.

# Set the timesteps 
Δt = 1seconds 
stop_time = 5hours
simulation = Simulation(model; Δt, stop_time)

# The TimeStepWizard helps ensure stable time-stepping with a default Courant-Freidrichs-Lewy (CFL) number of 0.7. 
conjure_time_step_wizard!(simulation)


# ## Track Simulation Progress 

# Wall clock represents the real world time as opposed to simulation time 
wall_clock = Ref(time_ns())

# Define callback function to log simulation iterations and time every 2 mins in simulation time 
function progress(sim)
    elapsed = 1e-9 * (time_ns() - wall_clock[])
    msg = @sprintf("Iter: %7d, time: %s, wall: %s, max|w| = %6.3e m s⁻¹",
                   iteration(sim),
                   prettytime(sim),
                   prettytime(elapsed),
                   maximum(abs, sim.model.velocities.w))
    wall_clock[] = time_ns()
    @info msg
    return nothing
end

save_interval = 2minutes
add_callback!(simulation, progress, name = :progress, TimeInterval(save_interval))


# ## Add Tracers and Diagnostics 

# Here we define the fields that we want to save a snapshot of at every 2minutes of simulation time. 
# The JLD2Writer saves: 
#  * b: buoyancy tracer
#  * e: turbulent kinetic energy tracer 
#  * u: x component velocity (horizontal) 
#  * u′: deviation of u from average of horizontal u
#  * w: z component velocity (vertical)
#  * N²: buoyancy stratification

b = model.tracers.b
e = model.tracers.e 
u, v, w = model.velocities

N² = ∂z(b)

filename = "lock_exchange.jld2"

simulation.output_writers[:fields] = JLD2Writer(model,
    (; b, e, u, w, N²);              
    filename = filename,
    schedule = TimeInterval(save_interval),
    overwrite_existing = true
)


# ## Run Simulation

run!(simulation)
@info "Simulation finished. Output saved to $(filename)"


# ## Load Saved TimeSeries Values 

ut  = FieldTimeSeries(filename, "u")
wt  = FieldTimeSeries(filename, "w")
N²t = FieldTimeSeries(filename, "N²")
bt = FieldTimeSeries(filename, "b")
et = FieldTimeSeries(filename, "e")
times = bt.times

@info "Saved times: $(times)"
@info "Number of snapshots: $(length(times))"


# ## Visualize Simulation Outputs 

# We use Makie's `Observable` to animate the data. To dive into how `Observable`s work we
# refer to [Makie.jl's Documentation](https://docs.makie.org/stable/explanations/observables).

n = Observable(1)

title = @lift @sprintf("t = %5.2f hours", times[$n] / hour)

uₙ = @lift ut[$n]
wₙ  = @lift wt[$n]
N²ₙ = @lift N²t[$n]
bₙ = @lift bt[$n]
eₙ = @lift et[$n]

# For visualization color ranges (use last snapshot)
umax = maximum(abs, ut[end])
wmax = maximum(abs, wt[end])
N2max = maximum(abs, N²t[end])
bmax = maximum(abs, bt[end])
emax = maximum(abs, et[end])
nothing #hide

# Use snapshots to create Makie visualization for b, N², u′, and w fields 

axis_kwargs = (xlabel = "x [m]",
               ylabel = "z [m]",
               limits = ((0, L), (-H, 0)),
               titlesize = 18)

fig = Figure(size = (800, 900))

fig[1, :] = Label(fig, title, fontsize = 24, tellwidth = false)

ax_b = Axis(fig[2, 1]; title = "b (Buoyancy)", axis_kwargs...)
hm_b = heatmap!(ax_b, bₙ; nan_color = :black, 
                colorrange = (0, bmax), colormap = :magma)
Colorbar(fig[2, 2], hm_b, label = "m s⁻²")

ax_e = Axis(fig[3, 1]; title = "e (turbulent kinetic energy)", axis_kwargs...)
hm_e = heatmap!(ax_e, eₙ; nan_color = :black,
                colorrange = (-0.25emax, emax), colormap = :magma)
Colorbar(fig[3, 2], hm_e, label = "m² s⁻²")

ax_u = Axis(fig[4, 1]; title = "u (horizontal velocity)", axis_kwargs...)
hm_u = heatmap!(ax_u, uₙ; nan_color = :black,
                colorrange = (-umax, umax), colormap = :magma)
Colorbar(fig[4, 2], hm_u, label = "m s⁻¹")

ax_N2 = Axis(fig[5, 1]; title = "N² (stratification)", axis_kwargs...)
hm_N2 = heatmap!(ax_N2, N²ₙ; nan_color = :black, 
                colorrange = (-0.25N2max, N2max), colormap = :magma)
Colorbar(fig[5, 2], hm_N2, label = "s⁻²")

fig


# ## Create Animation

@info "Making animation from saved simulation data..."

frames = 1:length(times)

record(fig, "lock_exchange.mp4", frames; framerate = 8) do i
    @info "Plotting frame $i / $(frames[end])"
    n[] = i
end

@info "Animation created in lock_exchange.mp4"


# The visualization shows the time evolution of buoyancy b, turbulent kinetic energy e, 
# stratification N²,and  velocity field u. 
# Initially, the two water masses are separated horizontally
# by a sharp density interface. As the flow evolves, gravity currents form and the dense
# fluid moves beneath the lighter fluid. This shows the characteristic transition from
# horizontal to vertical density separation for the lock exchange problem. 

# ![](lock_exchange.mp4)
