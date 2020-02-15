# # Simple diffusion example
#
# This script provides our simplest example of Oceananigans.jl functionality:
# the diffusion of a one-dimensional Gaussian. This example demonstrates
#
#   * how to load `Oceananigans.jl`;
#   * how to instantiate an `Oceananigans.jl` `Model`;
#   * how to set an initial condition with a function;
#   * how to time-step a model forward, and finally
#   * how to look at results.
#
# ## Using `Oceananigans.jl`
#
# To use `Oceananigans.jl` after it has been installed, we bring
# `Oceananigans.jl` functions and names into our 'namespace' by writing

using Oceananigans

# We also use `Plots.jl` for plotting and `Printf` to format plot legends:

using Plots, Printf

# ## Instantiating and configuring a `Model`
#
# To begin using Oceananigans, we instantiate a `Model` by calling the
# `Model` constructor:

model = Model(
    grid = RegularCartesianGrid(size = (1, 1, 128), x = (0, 1), y = (0, 1), z = (-0.5, 0.5)),
    closure = ConstantIsotropicDiffusivity(κ = 1.0)
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
Tᵢ(x, y, z) = exp( -z^2 / (2δ^2) )

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
# We use `Plots.jl` to look at the results. Tracers are defined at cell
# centers so we use `zC` as the z-coordinate when plotting it. Fields are
# stored as 3D arrays in Oceananigans so we plot `interior(T)[1, 1, :]`
# which will return a 1D array.

## A convenient function for generating a label with the current model time
tracer_label(model) = @sprintf("t = %.3f", model.clock.time)

## Plot initial condition
zC = model.grid.zC
p = plot(Tᵢ.(0, 0, zC), zC, linewidth=2, label="t = 0",
         xlabel="Tracer concentration", ylabel="z")

## Plot current solution
T = model.tracers.T
plot!(p, interior(T)[1, 1, :], zC, linewidth=2, label=tracer_label(model))

# Interesting! We can keep running the simulation and animate the tracer
# concentration to see the Gaussian diffusing.

anim = @animate for i=1:100
    simulation.stop_iteration += 100
    run!(simulation)

    plot(interior(T)[1, 1, :], zC, linewidth=2, title=tracer_label(model),
         label="", xlabel="Tracer concentration", ylabel="z", xlims=(0, 1))
end

mp4(anim, "1d_diffusion.mp4", fps = 15) # hide
