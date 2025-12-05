# # [Lock Exchange Example](@id lock_exchange_example)
#
# This is a simple example of a lock exchange problem on a slope. This example demonstrates: 
#
#  * How to set up a 2D grid with a sloping bottom with an immersed boundary
#  * Initializing a hydrostatic free surface model 
#  * Including variable density initial conditions 
#  * Saving outputs of the simulation 
#  * Creating an animation with CairoMakie


# ## Import Required Packages 

using Oceananigans
using Oceananigans.Units: kilometers, meters, seconds, minutes, days, hour
using Oceananigans.Grids: MutableVerticalDiscretization
using Oceananigans.Diagnostics: FieldTimeSeries, Average
using Oceananigans.OutputWriters
using Oceananigans.Operators

using Printf
using CairoMakie


# ## Set up 2D Rectilinear Grid

# Set resolution of the simulation grid 
Nx, Nz = 256, 64

# Set grid size 
L = 8kilometers   # horizontal length
H = 50meters      # depth  

# Allow for mutable surface height 
z_disc  = MutableVerticalDiscretization((-50, 0))
x = (0, L)

# Initialize the grid 
underlying_grid = RectilinearGrid(
                    size = (Nx, Nz),
                    halo = (5, 5),
                    x = x,
                    z = z_disc,
                    topology = (Bounded, Flat, Bounded)
)

# Add a slope at the bottom of the grid 
h_left = -H
h_right = -25meters
slope = (h_right - h_left) / L
bottom(x) = h_left + slope * x

# Use an immersed boundary with grid fitted bottom to describe the sloped bottom of the domain 
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom))


# ## Initialize the Model 
#
#  * Want to use a hydrostatic model since horizontal motion may be more significant than verticle motion 
#  * Tracers act as markers within the fluid to track movement and dispersion
#  * Weighted Essentially Non-Oscillatory (WENO) methods are useful for capturing sharp changes in density
#  * ZStarCoordinate method allows for the top of the grid to move with the free surface
#  * Runge Kutta method is good for integrating multiple processes 

model = HydrostaticFreeSurfaceModel(; grid,
    tracers = :b,
    buoyancy = BuoyancyTracer(),
    momentum_advection = WENO(order=5), 
    tracer_advection = WENO(order=7), 
    vertical_coordinate = ZStarCoordinate(grid), 
    free_surface = SplitExplicitFreeSurface(grid; substeps=10), 
    timestepper = :SplitRungeKutta3 
)

# ## Set Variable Density Initial Conditions 

# Set initial conditions for lock exchange with different densities 
bᵢ(x, z) = x > 4kilometers ? 0.06 : 0.01
set!(model, b=bᵢ)

# ## Defining Simulation Timestep 
# 
# Note that we have a small time step because of a small Courant-Friedrichs-Lewy (CLF) condition 

# Set the timesteps 
Δt = 1seconds 
stop_time = 1days
simulation = Simulation(model; Δt, stop_time)

# ## Track Simulation Progress 

wall_clock = Ref(time_ns())

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

save_interval = 30minutes
add_callback!(simulation, progress, name = :progress, TimeInterval(save_interval))


# ## Add Tracers and Diagnostics 

b = model.tracers.b
u, v, w = model.velocities

U = Field(Average(u)) 
u′ = u - U            

N² = ∂z(b)

filename = "lock_exchange.jld2"

simulation.output_writers[:fields] = JLD2Writer(model,
    (; b, u, u′, w, N²);              
    filename = filename,
    schedule = TimeInterval(save_interval),
    overwrite_existing = true
)

# ## Run Simulation

run!(simulation)
@info "Simulation finished. Output saved to $(filename)"

b_t = FieldTimeSeries("lock_exchange.jld2", "b")
times = b_t.times

@info "Saved times: $(times)"
@info "Number of snapshots: $(length(times))"


# ## Load Saved TimeSeries Values 

b_t  = FieldTimeSeries(filename, "b")
u_t  = FieldTimeSeries(filename, "u")
u′_t = FieldTimeSeries(filename, "u′")
w_t  = FieldTimeSeries(filename, "w")
N²_t = FieldTimeSeries(filename, "N²")

umax = maximum(abs, u′_t[end])
wmax = maximum(abs, w_t[end])


# ## Visualize Simulation Outputs 

n = Observable(1)

title = @lift @sprintf("t = %3.2f hours", times[$n] / hour)

u′ₙ = @lift u′_t[$n]
wₙ  = @lift w_t[$n]
N²ₙ = @lift N²_t[$n]


axis_kwargs = (xlabel = "x [m]",
               ylabel = "z [m]",
               limits = ((0, L), (-H, 0)),
               titlesize = 18)

fig = Figure(size = (800, 900))

fig[1, :] = Label(fig, title, fontsize = 24, tellwidth = false)

ax_u = Axis(fig[2, 1]; title = "u′ (along-slope velocity)", axis_kwargs...)
hm_u = heatmap!(ax_u, u′ₙ; nan_color = :black,
                colorrange = (-umax, umax), colormap = :balance)
Colorbar(fig[2, 2], hm_u, label = "m s⁻¹")

ax_w = Axis(fig[3, 1]; title = "w (vertical velocity)", axis_kwargs...)
hm_w = heatmap!(ax_w, wₙ; nan_color = :black,
                colorrange = (-wmax, wmax), colormap = :balance)
Colorbar(fig[3, 2], hm_w, label = "m s⁻¹")

ax_N2 = Axis(fig[4, 1]; title = "N² (stratification)", axis_kwargs...)
hm_N2 = heatmap!(ax_N2, N²ₙ; nan_color = :black, colormap = :magma)
Colorbar(fig[4, 2], hm_N2, label = "s⁻²")

display(fig)

# ### Create Animation

@info "Making animation from saved simulation data..."

frames = 1:length(times)

record(fig, "lock_exchange.mp4", frames; framerate = 10) do i
    @info "Plotting frame $i / $(frames[end])"
    n[] = i
end

@info "Animation created in lock_exchange.mp4"
