# # Simple diffusion example
#
# This script provides our simplest example of Oceananigans.jl functionality:
# the diffusion of a one-dimensional Gaussian. This example demonstrates
#
#   * how to load `Oceananigans.jl`;
#   * how to instantiate an `Oceananigans.jl` model;
#   * how to create simple `Oceananigans.jl` output;
#   * how to set an initial condition with a function;
#   * how to time-step a model forward, and finally
#   * how to look at results.
#
# ## Using `Oceananigans.jl`
#
# To use `Oceananigans.jl` after it has been installed, we bring
# `Oceananigans.jl` functions and names into our 'namespace' by writing

using Oceananigans

# In addition, we import the submodule `Grids`.

using Oceananigans.Grids

# ## Instantiating and configuring a model
#
# To begin using Oceananigans, we instantiate an incompressible model by calling
# the `IncompressibleModel` constructor:

model = IncompressibleModel(
       grid = RegularCartesianGrid(size = (1, 1, 128), x = (0, 1), y = (0, 1), z = (-0.5, 0.5)),
    closure = IsotropicDiffusivity(κ = 1.0)
)
nothing # hide

# The keyword arguments `grid` and `closure` indicate that
# our model grid is Cartesian with uniform grid spacing, that our diffusive
# stress and tracer fluxes are determined by diffusion with a constant
# diffusivity `κ` (note that we do not use viscosity in this example).

# Note that by default, a `Model` has no-flux boundary condition on all
# variables. Next, we set an initial condition on our "passive tracer",
# temperature. Our objective is to observe the diffusion of a Gaussian.

## Build a Gaussian initial condition function with width `δ`:
δ = 0.1
Tᵢ(x, y, z) = exp(-z^2 / (2δ^2))

## Set `model.tracers.T` to the function `Tᵢ`:
set!(model, T=Tᵢ)

# ## Running your first `Model`
#
# Finally, we time-step the model forward using the function
# `time_step!`, with a time-step size that ensures numerical stability.

## Time-scale for diffusion across a grid cell
cell_diffusion_time_scale = model.grid.Δz^2 / model.closure.κ.T

## We create a `Simulation` which will handle time stepping the model. It will
## execute `Nt` time steps with step size `Δt` using a second-order Adams-Bashforth method.
simulation = Simulation(model, Δt = 0.1 * cell_diffusion_time_scale, stop_iteration = 1000)

run!(simulation)

# ## Visualizing the results
#
# We use `Plots.jl` to look at the results. Fields are
# stored as 3D arrays in Oceananigans so we plot `interior(T)[1, 1, :]`
# which will return a 1D array.

using Plots, Printf

## A convenient function for generating a label with the current model time
tracer_label(time) = @sprintf("t = %.3f", time)

## Plot initial condition
T = model.tracers.T

z = znodes(T)[:]

p = plot(Tᵢ.(0, 0, z), z, linewidth=2, label="t = 0",
         xlabel="Temperature", ylabel="z")

## Plot current solution
plot!(p, interior(T)[1, 1, :], z, linewidth=2, label=tracer_label(model.clock.time))

# Interesting! Next, we add an output writer that saves the temperature field
# in JLD2 files, and run the simulation for longer so that we can animate the results.

using Oceananigans.OutputWriters: JLD2OutputWriter

simulation.output_writers[:temperature] =
    JLD2OutputWriter(model, model.tracers, prefix = "one_dimensional_diffusion",
                     iteration_interval = 100, force = true)

## Run simulation for 10,000 more iterations
simulation.stop_iteration += 10000

run!(simulation)
nothing

# Finally, we animate the results by opening the JLD2 file, extract the
# iterations we ended up saving at, and plot the evolution of the
# temperature profile in a loop over the iterations.

using JLD2

file = jldopen(simulation.output_writers[:temperature].filepath)

iterations = parse.(Int, keys(file["timeseries/t"]))

anim = @animate for (i, iter) in enumerate(iterations)

    T = file["timeseries/T/$iter"][1, 1, 2:end-1]
    t = file["timeseries/t/$iter"]

    plot(T, z, linewidth=2, title=tracer_label(t),
         label="", xlabel="Temperature", ylabel="z", xlims=(0, 1))
end

mp4(anim, "one_dimensional_diffusion.mp4", fps = 15) # hide
