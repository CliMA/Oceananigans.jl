# # Simple diffusion example
#
# This script provides our simplest example of Oceananigans.jl functionality:
# the diffusion of a one-dimensional Gaussian. This example demonstrates
#
#   * How to load `Oceananigans.jl`.
#   * How to instantiate an `Oceananigans.jl` model.
#   * How to create simple `Oceananigans.jl` output.
#   * How to set an initial condition with a function.
#   * How to time-step a model forward.
#   * How to look at results.
#
# ## Using `Oceananigans.jl`
#
# We write

using Oceananigans

# to load Oceananigans functions and objects into our script.
#
# ## Instantiating and configuring a model
#
# A core Oceananigans type is `IncompressibleModel`. We build an `IncompressibleModel`
# by passing it a `grid`, plus information about the equations we would like to solve:

model = IncompressibleModel(
       grid = RegularCartesianGrid(size = (1, 1, 128), x = (0, 1), y = (0, 1), z = (-0.5, 0.5)),
    closure = IsotropicDiffusivity(κ = 1.0)
)

# The `grid` and turbulence `closure` specify a Cartesian grid with regular
# (uniform) grid spacing, and isotropic tracer diffusion with
# constant diffusivity `κ`.
#
# Our simple `grid` and `model` use a number of defaults:
#
#   * The default `grid` topology periodic in `x, y` and bounded in `z`.
#   * The default `Model` has no-flux (insulating and stress-free) boundary conditions on
#     non-periodic boundaries for velocities `u, v, w` and tracers.
#   * The default `Model` has two tracers: temperature `T`, and salinity `S`.
#   * The default `Model` uses a `SeawaterBuoyancy` model with a `LinearEquationOfState`.
#     However, buoyancy will not be active in the simulation we run below.
#
# Next, we `set!` an initial condition on the temperature field,
# `model.tracers.T`. Our objective is to observe the diffusion of a Gaussian.

width = 0.1

initial_temperature(x, y, z) = exp(-z^2 / (2width^2))

set!(model, T=initial_temperature)

# ## Running a `Simulation`
#
# Next we set-up a `Simulation` that time-steps the model forward and manages output.

## Time-scale for diffusion across a grid cell
diffusion_time_scale = model.grid.Δz^2 / model.closure.κ.T

simulation = Simulation(model, Δt = 0.1 * diffusion_time_scale, stop_iteration = 1000)

# We've specified that `simulation` runs for 1000 iterations with a stable time-step.
# All that's left to do is

run!(simulation)

# ## Visualizing the results
#
# We use `Plots.jl` to look at the results.

using Plots, Printf

using Oceananigans.Grids: znodes ## for obtaining z-coordinates of Oceananigans fields

# We plot the initial condition and the current solution. Note that
# fields are always 3D in Oceananigans. We use `interior(model.tracers.T)[1, 1, :]`
# to plot temperature, which returns a 1D array of `z`-values.

## A convenient function for generating a label with the current model time
time_label(time) = @sprintf("t = %.3f", time)

z = znodes(model.tracers.T)

p = plot(initial_temperature.(0, 0, z), z, linewidth=2, label="t = 0", xlabel="Temperature", ylabel="z")

plot!(p, interior(model.tracers.T)[1, 1, :], z, linewidth=2, label=time_label(model.clock.time))

# Very interesting! Next, we run the simulation a bit longer and make an animation.
# For this, we use the `JLD2OutputWriter` to write data to disk as the simulation progresses.

using Oceananigans.OutputWriters: JLD2OutputWriter, IterationInterval

simulation.output_writers[:temperature] =
    JLD2OutputWriter(model, model.tracers, prefix = "one_dimensional_diffusion",
                     schedule=IterationInterval(100), force = true)

# We run the simulation for 10,000 more iterations,

simulation.stop_iteration += 10000

run!(simulation)

# Finally, we animate the results by opening the JLD2 file, extract the
# iterations we ended up saving at, and plot the evolution of the
# temperature profile in a loop over the iterations.

using JLD2

file = jldopen(simulation.output_writers[:temperature].filepath)

iterations = parse.(Int, keys(file["timeseries/t"]))

anim = @animate for (i, iter) in enumerate(iterations)

    T = file["timeseries/T/$iter"][1, 1, :]
    t = file["timeseries/t/$iter"]

    plot(T, z, linewidth=2, title=time_label(t),
         label="", xlabel="Temperature", ylabel="z", xlims=(0, 1))
end

mp4(anim, "one_dimensional_diffusion.mp4", fps = 15) # hide
