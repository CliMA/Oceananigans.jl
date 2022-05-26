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
# pkg"add Oceananigans, CairoMakie"
# ```

# ## Using `Oceananigans.jl`
#
# Write

using Oceananigans

# to load Oceananigans functions and objects into our script.
#
# ## Instantiating and configuring a model
#
# A core Oceananigans type is `NonhydrostaticModel`. We build a `NonhydrostaticModel`
# by passing it a `grid`, plus information about the equations we would like to solve.
#
# Below, we build a rectilinear grid with 128 regularly-spaced grid points in
# the `z`-direction, where `z` spans from `z = -0.5` to `z = 0.5`,

grid = RectilinearGrid(size=128, z=(-0.5, 0.5), topology=(Flat, Flat, Bounded))

# The default topology is `(Periodic, Periodic, Bounded)`. In this example we're
# trying to solve a one-dimensional problem, so we assign `Flat` to the
# `x` and `y` topologies. We excise halos and avoid interpolation or differencing
# in `Flat` directions, saving computation and memory.
#
# We next specify a model with an `ScalarDiffusivity`, which models either
# molecular or turbulent diffusion,

closure = ScalarDiffusivity(κ=1)

# We finally pass these two ingredients to `NonhydrostaticModel`,

model = NonhydrostaticModel(grid=grid, closure=closure, buoyancy=nothing, tracers=:T)

# By default, `NonhydrostaticModel` has no-flux (insulating and stress-free) boundary conditions on
# all fields.
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

using CairoMakie

linewidth = 3
z = znodes(model.tracers.T)

fig = Figure()

lines(interior(model.tracers.T, 1, 1, :), z; linewidth, label = "t = 0",
     axis = (xlabel = "Temperature (ᵒC)",
             ylabel = "z",
             xlabelsize = 20,
             ylabelsize = 20))

# The function `interior` above extracts a `view` of `model.tracers.T` over the
# physical points (excluding halos) at `(1, 1, :)`.
#
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

label = @sprintf("t = %.3f", model.clock.time)

lines!(interior(model.tracers.T, 1, 1, :), z; linewidth, label)

axislegend()

# Very interesting! Next, we run the simulation a bit longer and make an animation.
# For this, we use the `JLD2OutputWriter` to write data to disk as the simulation progresses.

using Oceananigans.OutputWriters: JLD2OutputWriter, IterationInterval

simulation.output_writers[:temperature] =
    JLD2OutputWriter(model, model.tracers,
                     filename = "one_dimensional_diffusion.jld2",
                     schedule=IterationInterval(100),
                     overwrite_existing = true)

# We run the simulation for 10,000 more iterations,

simulation.stop_iteration += 10000

run!(simulation)

# Finally, we animate the results by opening the JLD2 file, extract the
# iterations we ended up saving at, and plot the evolution of the
# temperature profile in a loop over the iterations.

T_timeseries = FieldTimeSeries("one_dimensional_diffusion.jld2", "T")
times = T_timeseries.times

n = Observable(1)

T = @lift interior(T_timeseries[$n], 1, 1, :)

label = @lift "t = " * string(round(times[$n], digits=3))

fig = Figure()

ax = Axis(fig[2, 1];
          xlabel = "Temperature (ᵒC)",
          ylabel = "z",
          xlabelsize = 20,
          ylabelsize = 20,
          limits = ((0, 1), nothing))

lines!(T, z; linewidth)

fig[1, 1] = Label(fig, label, textsize=24, tellwidth=false)

# Finally, we record a movie.

frames = 1:length(times)

record(fig, "one_dimensional_diffusion.mp4", frames, framerate=8) do i
    @info "Plotting frame $i of $(frames[end])..."
    n[] = i
end
nothing #hide

# ![](one_dimensional_diffusion.mp4)
