# # [Simple diffusion example](@id one_dimensional_diffusion_example)
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

# The default topology is `(Periodic, Periodic, Bounded)`. In this example, we're
# trying to solve a one-dimensional problem, so we assign `Flat` to the
# `x` and `y` topologies. We excise halos and avoid interpolation or differencing
# in `Flat` directions, saving computation and memory.
#
# We next specify a `ScalarDiffusivity` with diffusivity ``κ``, which models either
# molecular or turbulent diffusion,

κ = 1
closure = ScalarDiffusivity(; κ)

# We finally pass these two ingredients to `NonhydrostaticModel`,

model = NonhydrostaticModel(grid; closure, tracers=:T)

# By default, `NonhydrostaticModel` has no-flux (insulating and stress-free) boundary conditions on
# all fields.
#
# Next, we `set!` an initial condition on the temperature field,
# `model.tracers.T`. Our objective is to observe the diffusion of a Gaussian.

width = 0.1
initial_temperature(z) = exp(-z^2 / (2width^2))
set!(model, T=initial_temperature)

# ## Visualizing model data
#
# Calling `set!` above changes the data contained in `model.tracers.T`,
# which was initialized as `0`'s when the model was created.
# To see the new data in `model.tracers.T`, we plot it:

using CairoMakie
set_theme!(Theme(fontsize = 20, linewidth=3))

axis = (xlabel = "Temperature (ᵒC)", ylabel = "z")
label = "t = 0"
lines(model.tracers.T; label, axis)
current_figure() #hide

# ## Running a `Simulation`
#
# Next we set up a `Simulation` that time-steps the model forward and manages output.

Δz = minimum_zspacing(grid)
Δt = 0.1 * Δz^2 / κ

simulation = Simulation(model; Δt, stop_iteration = 1000)

# `simulation` will run for 1000 iterations with a time-step that resolves the time-scale
# ``Δz^2 / κ`` for diffusion across a grid cell. All that's left is to

## Fail the docs build if this simulation produces NaNs #hide
Oceananigans.Diagnostics.erroring_NaNChecker!(simulation) #hide
run!(simulation)

# ## Visualizing the results
#
# Let's look at how `model.tracers.T` changed during the simulation.

label = "t = $(round(model.clock.time, digits=3))"
lines!(model.tracers.T; label)
axislegend()
current_figure() #hide

# Very interesting! Next, we run the simulation a bit longer and make an animation.
# For this, we use the `JLD2Writer` to write data to disk as the simulation progresses.

simulation.output_writers[:temperature] =
    JLD2Writer(model, model.tracers,
               filename = "one_dimensional_diffusion.jld2",
               schedule = IterationInterval(100),
               overwrite_files = true)

# We run the simulation for 10,000 more iterations,

simulation.stop_iteration += 10000
run!(simulation)

# To animate the results, we load the saved temperature as a `FieldTimeSeries`
# and plot the temperature profile at each saved time.

T_timeseries = FieldTimeSeries("one_dimensional_diffusion.jld2", "T")
times = T_timeseries.times

fig = Figure()
ax = Axis(fig[2, 1]; xlabel = "Temperature (ᵒC)", ylabel = "z")
xlims!(ax, 0, 1)

n = Observable(1)

T = @lift T_timeseries[$n]
lines!(ax, T)

label = @lift "t = $(round(times[$n], digits=3))"
Label(fig[1, 1], label, tellwidth=false)

fig

# Finally, we record a movie.

frames = 1:length(times)

@info "Making an animation..."

record(fig, "one_dimensional_diffusion.mp4", frames, framerate=24) do i
    n[] = i
end
nothing #hide

# ![](one_dimensional_diffusion.mp4)
