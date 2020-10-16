# # Two dimensional turbulence example
#
# In this example, we initialize a random velocity field and observe its turbulent decay 
# in a two-dimensional domain. This example demonstrates:
#
#   * How to run a model with no buoyancy equation or tracers;
#   * How to create user-defined fields
#   * How to use differentiation functions
#   * How to save a computed field

# ## Model setup

# We instantiate the model with a simple isotropic diffusivity. We also use a 4-th order 
# advection scheme and Runge-Kutta 3rd order time-stepping scheme.

using Oceananigans, Oceananigans.Advection

model = IncompressibleModel(
        grid = RegularCartesianGrid(size=(128, 128, 1), halo=(2, 2, 2), extent=(2π, 2π, 2π)),
 timestepper = :RungeKutta3, 
   advection = CenteredFourthOrder(),
    buoyancy = nothing,
     tracers = nothing,
     closure = IsotropicDiffusivity(ν=1e-4)
)
nothing # hide

# ## Setting initial conditions

# Our initial condition randomizes `u` and `v`. We also ensure that both have
# zero mean for purely aesthetic reasons.

using Statistics

u₀ = rand(size(model.grid)...)
u₀ .-= mean(u₀)

set!(model, u=u₀, v=u₀)

# ## Calculating vorticity

using Oceananigans.Fields, Oceananigans.AbstractOperations

# Next we create an object called an `ComputedField` that calculates vorticity. We'll use
# this object to calculate vorticity on-line and output it as the simulation progresses.

u, v, w = model.velocities

ω = ComputedField(∂x(v) - ∂y(u))
nothing # hide

# Now we construct a simulation.

simulation = Simulation(model, Δt=0.1, stop_iteration=2000)

# ## Output
#
# We set up an output writer for the simulation that saves the vorticity every 20 iterations.

using Oceananigans.OutputWriters

simulation.output_writers[:fields] =
    JLD2OutputWriter(model, (ω = ω,), schedule=IterationInterval(20),
                     prefix = "2d_turbulence_vorticity", force = true)

# ## Running the simulation
#
# Finally, we run the simulation.

run!(simulation)

# # Visualizing the results
#
# We load the output and make a movie.

using JLD2

file = jldopen(simulation.output_writers[:fields].filepath)

iterations = parse.(Int, keys(file["timeseries/t"]))
nothing # hide

# Construct the $x$, $y$ grid for plotting purposes,

using Oceananigans.Grids

x, y = xnodes(ω)[:], ynodes(ω)[:]
nothing # hide

# and animate the vorticity.

using Plots

anim = @animate for iteration in iterations
    
    ω_snapshot = file["timeseries/ω/$iteration"][:, :, 1]

    heatmap(x, y, ω_snapshot',
            xlabel="x", ylabel="y",
            aspectratio=1,
            color=:balance,
            clims=(-0.2, 0.2),
            xlims=(0, model.grid.Lx),
            ylims=(0, model.grid.Ly))
end

mp4(anim, "2d_turbulence_vorticity.mp4", fps = 15) # hide
