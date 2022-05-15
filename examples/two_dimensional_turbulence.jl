# # Two dimensional turbulence example
#
# In this example, we initialize a random velocity field and observe its turbulent decay
# in a two-dimensional domain. This example demonstrates:
#
#   * How to run a model with no tracers and no buoyancy model.
#   * How to use `AbstractOperations`.
#   * How to use computed `Field`s to generate output.

# ## Install dependencies
#
# First let's make sure we have all required packages installed.

# ```julia
# using Pkg
# pkg"add Oceananigans, Plots"
# ```

# ## Model setup

# We instantiate the model with an isotropic diffusivity. We use a grid with 128² points,
# a fifth-order advection scheme, third-order Runge-Kutta time-stepping,
# and a small isotropic viscosity.  Note that we assign `Flat` to the `z` direction.

using Oceananigans

grid = RectilinearGrid(size=(128, 128), extent=(2π, 2π), 
                       topology=(Periodic, Periodic, Flat))

model = NonhydrostaticModel(timestepper = :RungeKutta3,
                              advection = UpwindBiasedFifthOrder(),
                                   grid = grid,
                               buoyancy = nothing,
                                tracers = nothing,
                                closure = ScalarDiffusivity(ν=1e-5)
                           )

# ## Random initial conditions
#
# Our initial condition randomizes `model.velocities.u` and `model.velocities.v`.
# We ensure that both have zero mean for aesthetic reasons.

using Statistics

u, v, w = model.velocities

uᵢ = rand(size(u)...)
vᵢ = rand(size(v)...)

uᵢ .-= mean(uᵢ)
vᵢ .-= mean(vᵢ)

set!(model, u=uᵢ, v=vᵢ)

simulation = Simulation(model, Δt=0.2, stop_time=50)

# ## Logging simulation progress
#
# We set up a callback that logs the simulation iteration and time every 100 iterations.

progress(sim) = @info "Iteration: $(iteration(sim)), time: $(time(sim))"

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

# ## Output
#
# We set up an output writer for the simulation that saves vorticity and speed every 20 iterations.
#
# ### Computing vorticity and speed
#
# To make our equations prettier, we unpack `u`, `v`, and `w` from
# the `NamedTuple` model.velocities:
u, v, w = model.velocities

# Next we create two `Field`s that calculate
# _(i)_ vorticity that measures the rate at which the fluid rotates
# and is defined as
#
# ```math
# ω = ∂_x v - ∂_y u \, ,
# ```

ω = ∂x(v) - ∂y(u)

# We also calculate _(ii)_ the _speed_ of the flow,
#
# ```math
# s = \sqrt{u^2 + v^2} \, .
# ```

s = sqrt(u^2 + v^2)

# We pass these operations to an output writer below to calculate and output them during the simulation.
filename = "two_dimensional_turbulence"

simulation.output_writers[:fields] = JLD2OutputWriter(model, (; ω, s),
                                                      schedule = TimeInterval(2),
                                                      filename = filename * ".jld2",
                                                      overwrite_existing = true)

# ## Running the simulation
#
# Pretty much just

run!(simulation)

# ## Visualizing the results
#
# We load the output.

ω_timeseries = FieldTimeSeries(filename * ".jld2", "ω")
s_timeseries = FieldTimeSeries(filename * ".jld2", "s")

times = ω_timeseries.times

# Construct the ``x, y, z`` grid for plotting purposes,

xω, yω, zω = nodes(ω_timeseries)
xs, ys, zs = nodes(s_timeseries)
nothing # hide

# and animate the vorticity and fluid speed.

using CairoMakie

@info "Making a neat movie of vorticity and speed..."

fig = Figure(resolution = (800, 400))

axis_kwargs = (xlabel = "x",
               ylabel = "y",
               titlesize = 24,
               limits = ((0, 2π), (0, 2π)),
               aspect = AxisAspect(1))

ax_ω = Axis(fig[2, 1]; title = "vorticity", axis_kwargs...)
ax_s = Axis(fig[2, 2]; title = "speed", axis_kwargs...)

nothing #hide

# We use Makie's `Observable` to animate the data. To dive into how `Observable`s work we
# refer to [Makie.jl's Documentation](https://makie.juliaplots.org/stable/documentation/nodes/index.html).

n = Observable(1)

title = @lift "t = " * string(round(times[$n], digits=2))

ω = @lift interior(ω_timeseries[$n], :, :, 1)
s = @lift interior(s_timeseries[$n], :, :, 1)

# Now let's plot the vorticity and speed.

ω_lim = 2.0

heatmap!(ax_ω, xω, yω, ω;
         colormap = :balance, colorrange = (-ω_lim, ω_lim))

s_lim = 0.2

heatmap!(ax_s, xs, ys, s;
         colormap = :speed, colorrange = (0, s_lim))

fig[1, :] = Label(fig, title, textsize=24, tellwidth=false)

# Finally, we record a movie.

frames = 1:length(times)

record(fig, filename * ".mp4", frames, framerate=8) do frame
       @info "Plotting frame $i of $(frames[end])..."
       n[] = frame
end
nothing #hide

# ![](two_dimensional_turbulence.mp4)
