# # Two dimensional turbulence example
#
# In this example, we initialize a random velocity field and observe its turbulent decay
# in a two-dimensional domain. This example demonstrates:
#
#   * How to run a model with no tracers and no buoyancy model.
#   * How to use `AbstractOperations`.
#   * How to use `ComputedField`s to generate output.

# ## Install dependencies
#
# First let's make sure we have all required packages installed.

# ```julia
# using Pkg
# pkg"add Oceananigans, JLD2, Plots"
# ```

# ## Model setup

# We instantiate the model with an isotropic diffusivity. We use a grid with 128² points,
# a fifth-order advection scheme, third-order Runge-Kutta time-stepping,
# and a small isotropic viscosity.  Note that we assign `Flat` to the `z` direction.

using Oceananigans

grid = RegularRectilinearGrid(size=(128, 128), extent=(2π, 2π), 
                              topology=(Periodic, Periodic, Flat))

model = NonhydrostaticModel(timestepper = :RungeKutta3,
                              advection = UpwindBiasedFifthOrder(),
                                   grid = grid,
                               buoyancy = nothing,
                                tracers = nothing,
                                closure = IsotropicDiffusivity(ν=1e-5)
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

# ## Computing vorticity and speed

# To make our equations prettier, we unpack `u`, `v`, and `w` from
# the `NamedTuple` model.velocities:
u, v, w = model.velocities

# Next we create two objects called `ComputedField`s that calculate
# _(i)_ vorticity that measures the rate at which the fluid rotates
# and is defined as
#
# ```math
# ω = ∂_x v - ∂_y u \, ,
# ```

ω = ∂x(v) - ∂y(u)

ω_field = ComputedField(ω)

# We also calculate _(ii)_ the _speed_ of the flow,
#
# ```math
# s = \sqrt{u^2 + v^2} \, .
# ```

s = sqrt(u^2 + v^2)

s_field = ComputedField(s)

# We'll pass these `ComputedField`s to an output writer below to calculate and output them during the simulation.

simulation = Simulation(model, Δt=0.2, stop_time=50)

# ## Logging simulation progress
#
# We set up a callback that logs the simulation iteration and time every 100 iterations.

progress(sim) = @info "Iteration: $(iteration(sim)), time: $(time(sim))"

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

# ## Output
#
# We set up an output writer for the simulation that saves the vorticity every 20 iterations.

simulation.output_writers[:fields] = JLD2OutputWriter(model, (ω=ω_field, s=s_field),
                                                      schedule = TimeInterval(2),
                                                      prefix = "two_dimensional_turbulence",
                                                      force = true)

# ## Running the simulation
#
# Pretty much just

run!(simulation)

# ## Visualizing the results
#
# We load the output and make a movie.

using JLD2

file = jldopen(simulation.output_writers[:fields].filepath)

iterations = parse.(Int, keys(file["timeseries/t"]))

# Construct the ``x, y`` grid for plotting purposes,

xω, yω, zω = nodes(ω_field)
xs, ys, zs = nodes(s_field)
nothing # hide

# and animate the vorticity and fluid speed.

using Plots

@info "Making a neat movie of vorticity and speed..."

anim = @animate for (i, iteration) in enumerate(iterations)

    @info "Plotting frame $i from iteration $iteration..."

    t = file["timeseries/t/$iteration"]
    ω_snapshot = file["timeseries/ω/$iteration"][:, :, 1]
    s_snapshot = file["timeseries/s/$iteration"][:, :, 1]

    ω_lim = 2.0
    ω_levels = range(-ω_lim, stop=ω_lim, length=20)

    s_lim = 0.2
    s_levels = range(0, stop=s_lim, length=20)

    kwargs = (xlabel="x", ylabel="y", aspectratio=1, linewidth=0, colorbar=true,
              xlims=(0, model.grid.Lx), ylims=(0, model.grid.Ly))

    ω_plot = contourf(xω, yω, clamp.(ω_snapshot', -ω_lim, ω_lim);
                       color = :balance,
                      levels = ω_levels,
                       clims = (-ω_lim, ω_lim),
                      kwargs...)

    s_plot = contourf(xs, ys, clamp.(s_snapshot', 0, s_lim);
                       color = :thermal,
                      levels = s_levels,
                       clims = (0., s_lim),
                      kwargs...)

    plot(ω_plot, s_plot, title=["Vorticity" "Speed"], layout=(1, 2), size=(1200, 500))
end

mp4(anim, "two_dimensional_turbulence.mp4", fps = 8) # hide
