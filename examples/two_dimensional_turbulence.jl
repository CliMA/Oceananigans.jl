# # Two dimensional turbulence example
#
# In this example, we initialize a random velocity field and observe its turbulent decay 
# in a two-dimensional domain. This example demonstrates:
#
#   * How to run a model with no tracers and no buoyancy model.
#   * How to use `AbstractOperations`.
#   * How to use `ComputedField`s to generate output.

# ## Model setup

# We instantiate the model with an isotropic diffusivity. We use a grid with 128² points,
# a fifth-order advection scheme, third-order Runge-Kutta time-stepping,
# and a small isotropic viscosity.

using Oceananigans, Oceananigans.Advection

grid = RegularCartesianGrid(size=(128, 128, 1), extent=(2π, 2π, 2π))

model = IncompressibleModel(timestepper = :RungeKutta3, 
                              advection = UpwindBiasedFifthOrder(),
                                   grid = grid,
                               buoyancy = nothing,
                                tracers = nothing,
                                closure = IsotropicDiffusivity(ν=1e-5)
                           )

# ## Setting initial conditions

# Our initial condition randomizes `model.velocities.u` and `model.velocities.v`.
# We ensure that both have zero mean for aesthetic reasons.

using Statistics

u₀ = rand(size(model.grid)...)
u₀ .-= mean(u₀)

set!(model, u=u₀, v=u₀)

# ## Calculating vorticity

using Oceananigans.Fields, Oceananigans.AbstractOperations

# Next we create two objects called `ComputedField`s that calculate
# _(i)_ vorticity, and _(ii)_ speed. We'll use them to output vorticity
# and speed while the simulation runs.

u, v, w = model.velocities

vorticity = ComputedField(∂x(v) - ∂y(u))

speed = ComputedField(sqrt(u^2 + v^2))

# Now we construct a simulation that prints out the iteration and model time as it runs.

progress(sim) = @info "Iteration: $(sim.model.clock.iteration), time: $(round(Int, sim.model.clock.time))"

simulation = Simulation(model, Δt=0.2, stop_time=50, iteration_interval=100, progress=progress)

# ## Output
#
# We set up an output writer for the simulation that saves the vorticity every 20 iterations.

using Oceananigans.OutputWriters

simulation.output_writers[:fields] = JLD2OutputWriter(model, (ω=vorticity, s=speed),
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

using Oceananigans.Grids

xω, yω, zω = nodes(vorticity)
xs, ys, zs = nodes(speed)

# and animate the vorticity.

using Plots

@info "Making a neat movie of vorticity and speed..."

anim = @animate for (i, iteration) in enumerate(iterations)

    @info "Plotting frame $i from iteration $iteration..."
    
    t = file["timeseries/t/$iteration"]
    ω = file["timeseries/ω/$iteration"][:, :, 1]
    s = file["timeseries/s/$iteration"][:, :, 1]

    ω_max = maximum(abs, ω) + 1e-9
    ω_lim = 2.0

    s_max = maximum(abs, s) + 1e-9
    s_lim = 0.2

    ω_levels = vcat([-ω_max], range(-ω_lim, stop=ω_lim, length=20), [ω_max])
    s_levels = vcat(range(0, stop=s_lim, length=20), [s_max]) 

    kwargs = (xlabel="x", ylabel="y", aspectratio=1, linewidth=0, colorbar=true,
              xlims=(0, model.grid.Lx), ylims=(0, model.grid.Ly))

    ω_plot = contourf(xω, yω, ω';
                       color = :balance,
                      levels = ω_levels,
                       clims = (-ω_lim, ω_lim),
                      kwargs...)

    s_plot = contourf(xs, ys, s';
                       color = :thermal,
                      levels = s_levels,
                       clims = (0, s_lim),
                      kwargs...)

    plot(ω_plot, s_plot, title=["Vorticity" "Speed"], layout=(1, 2), size=(1200, 500))
end

mp4(anim, "two_dimensional_turbulence.mp4", fps = 8) # hide
