# # Simple diffusion example
#
# This is Oceananigans.jl's simplest example:
# the diffusion of a one-dimensional Gaussian. This example demonstrates
#
#   * How to load `Oceananigans.jl`.
#   * How to instantiate an `Oceananigans.jl` model.
#   * How to create simple `Oceananigans.jl` output.
#   * How to set an initial condition with a function.
#   * How to time-step a model forward.
#   * How to look at results.
#
# ## Install dependencies
#
# First let's make sure we have all required packages installed.

# ```julia
# using Pkg
# pkg"add Oceananigans, JLD2, Plots"
# ```

# ## Using `Oceananigans.jl`
#
# Write

using Oceananigans

# to load Oceananigans functions and objects into our script.
#
# ## Instantiating and configuring a model
#
# A core Oceananigans type is `NonhydrostaticModel`. We build an `NonhydrostaticModel`
# by passing it a `grid`, plus information about the equations we would like to solve.
#
# Below, we build a regular rectilinear grid with 128 grid points in the `z`-direction,
# where `z` spans from `z = -0.5` to `z = 0.5`,

grid = RectilinearGrid(size=128, z=(-0.5, 0.5), topology=(Flat, Flat, Bounded))

# The default topology is `(Periodic, Periodic, Bounded)` but since we only want to solve
# a one-dimensional problem, we assign the `x` and `y` dimensions to `Flat`.  
# We could specify each of them to be either `Periodic` or `Bounded` but that will define
# a halo in each of those directions, and that is numerically more costly.  
# Note that we only specify the extent and size for the `Bounded` dimension.
#
# We next specify a model with an `IsotropicDiffusivity`, which models either
# molecular or turbulent diffusion,

closure = IsotropicDiffusivity(κ=1.0)

# We finally pass these two ingredients to `NonhydrostaticModel`,

model = NonhydrostaticModel(grid=grid, closure=closure)

# Our simple `grid` and `model` use a number of defaults:
#
#   * The default `grid` topology is periodic in `x, y` and bounded in `z`.
#   * The default `Model` has no-flux (insulating and stress-free) boundary conditions on
#     non-periodic boundaries for velocities `u, v, w` and tracers.
#   * The default `Model` has two tracers: temperature `T`, and salinity `S`.
#   * The default `Model` uses a `SeawaterBuoyancy` model with a `LinearEquationOfState`.
#     However, buoyancy is not active in the simulation we run below.
#
# Next, we `set!` an initial condition on the temperature field,
# `model.tracers.T`. Our objective is to observe the diffusion of a Gaussian.

width = 0.1

initial_temperature(x, y, z) = exp(-z^2 / (2width^2))

set!(model, T=initial_temperature)

# ## Visualizing model data
#
# Calling `set!` above changes the data contained in `model.tracers.T`,
# which was initialized as `0`'s when the model was created.
# To see the new data in `model.tracers.T`, we plot it:

using Plots

z = znodes(model.tracers.T)

T_plot = plot(interior(model.tracers.T)[1, 1, :], z,
              linewidth = 2,
              label = "t = 0",
              xlabel = "Temperature (ᵒC)",
              ylabel = "z")

# The function `interior` above extracts a `view` of the physical interior points
# of `model.tracers.T`. This is useful because `model.tracers.T` also contains "halo" points
# that lie outside the physical domain (halo points are used to set boundary conditions
# during time-stepping).

# ## Running a `Simulation`
#
# Next we set-up a `Simulation` that time-steps the model forward and manages output.

## Time-scale for diffusion across a grid cell
diffusion_time_scale = model.grid.Δzᵃᵃᶜ^2 / model.closure.κ.T

simulation = Simulation(model, Δt = 0.1 * diffusion_time_scale, stop_iteration = 1000)

# `simulation` will run for 1000 iterations with a time-step that resolves the time-scale
# at which our temperature field diffuses. All that's left is to

run!(simulation)

# ## Visualizing the results
#
# Let's look at how `model.tracers.T` changed during the simulation.

using Printf

plot!(T_plot, interior(model.tracers.T)[1, 1, :], z, linewidth=2,
      label=@sprintf("t = %.3f", model.clock.time))

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

    plot(T, z, linewidth=2, title=@sprintf("t = %.3f", t),
         label="", xlabel="Temperature", ylabel="z", xlims=(0, 1))
end

mp4(anim, "one_dimensional_diffusion.mp4", fps = 15) # hide
