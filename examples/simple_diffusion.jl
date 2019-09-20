using Oceananigans, PyPlot, Printf

# This script provides our simplest example of Oceananigans.jl functionality:
# the diffusion of a one-dimensional Gaussian. This example demonstrates:
#
#   * How to instantiate an `Oceananigans.jl` `Model`;
#   * How to set an initial condition with a function;
#   * How to time-step a model forward, and finally
#   * How to look at results.
#
# To begin, we instantiate a simple model on a `RegularCartesianGrid`, 
# with a constant diffusivity and no buoyancy forces:

model = Model(
    grid = RegularCartesianGrid(N = (1, 1, 128), L = (1, 1, 1)),
    closure = ConstantIsotropicDiffusivity(κ = 1.0),
    buoyancy = nothing,
)

# Note that by default, our model has no-flux boundary condition on all
# variables. Next, we set an initial condition on our "passive tracer", 
# temperature. Our objective is to observe the diffusion of a Gaussian.

Tᵢ(x, y, z) = exp( -(z + 0.5)^2 / 0.02 )
set!(model, T=Tᵢ)

# We time-step the model forwardFinally, we timNote that by default, 
# our model has no-flux boundary condition on all
# variables. Next, we set an initial condition on temperature. 
# Our objective is to observe the diffusion of a Gaussian.

## Time-scale for diffusion across a grid cell:
cell_diffusion_time_scale = model.grid.Δz^2 / model.closure.κ
time_step!(model, Nt = 1000, Δt = 0.1 * cell_diffusion_time_scale)

# Finally, we look at the results:

## Current model time
tracer_label(model) = @sprintf("\$ t=%.3f \$", model.clock.time)

## Create a plot
close("all")
fig, ax = subplots()
xlabel("Tracer")
ylabel(L"z")

## Plot initial condition
plot(Tᵢ.(0, 0, model.grid.zC), model.grid.zC, "-", label=L"t=0")

## Plot current solution
plot(data(model.tracers.T)[1, 1, :], model.grid.zC, "--", label=tracer_label(model))
legend()

# Interesting! Running the model even longer makes even more interesting results:

for i = 1:3
    time_step!(model, Nt = 1000, Δt = 0.1 * cell_diffusion_time_scale)
    plot(data(model.tracers.T)[1, 1, :], model.grid.zC, "--", label=tracer_label(model))
end

legend()
