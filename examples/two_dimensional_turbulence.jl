# # Two dimensional turbulence example
#
# In this example, we initialize a random velocity field and observe its viscous,
# turbulent decay in a two-dimensional domain. This example demonstrates:
#
#   * How to run a model with no buoyancy equation or tracers;
#   * How to create user-defined fields
#   * How to use differentiation functions

# ## Model setup

# For this example, we need `Plots` for plotting and `Statistics` for setting up
# a random initial condition with zero mean velocity.

using Oceananigans, Oceananigans.AbstractOperations
using Plots, Statistics

# In addition to importing plotting and statistics packages, we import
# some types from `Oceananigans` that will aid in the calculation
# and visualization of voriticty.

using Oceananigans: Face, Cell

# `Face` and `Cell` represent "locations" on the staggered grid. We instantiate the
# model with a simple isotropic diffusivity.

model = Model(
        grid = RegularCartesianGrid(size=(128, 128, 1), length=(2π, 2π, 2π)),
    buoyancy = nothing,
     tracers = nothing,
     closure = ConstantIsotropicDiffusivity(ν=1e-3, κ=1e-3)
)
nothing # hide

# ## Setting initial conditions

# Our initial condition randomizes `u` and `v`. We also ensure that both have
# zero mean for purely aesthetic reasons.

u₀ = rand(size(model.grid)...)
u₀ .-= mean(u₀)

set!(model, u=u₀, v=u₀)

# ## Calculating vorticity

# Next we create an object called an `Operation` that represents a vorticity calculation.
# We'll use this object to calculate vorticity on-line as the simulation progresses.

u, v, w = model.velocities
nothing # hide

# Create an object that represents the 'operation' required to compute vorticity.
vorticity_operation = ∂x(v) - ∂y(u)
nothing # hide

# The instance `vorticity_operation` is a binary subtraction between two derivative operations
# acting on `OffsetArrays` (the underyling representation of `u`, and `v`). In order to use
# `vorticity_operation` we create a field `ω` to store the result of the operation, and a
# `Computation` object for coordinate the computation of vorticity and storage in `ω`:

ω = Field(Face, Face, Cell, model.architecture, model.grid)

vorticity_computation = Computation(vorticity_operation, ω)
nothing # hide

# We ask for computation of vorticity by writing `compute!(vorticity_computation)`
# as shown below.

# ## Visualizing the simulation

# Finally, we run the model and animate the vorticity field.
simulation = Simulation(model, Δt=0.1, stop_iteration=0)

anim = @animate for i=1:100
    simulation.stop_iteration += 10
    run!(simulation)

    compute!(vorticity_computation)

    x, y = model.grid.xF, model.grid.yF
    heatmap(x, y, interior(ω)[:, :, 1], xlabel="x", ylabel="y",
            color=:balance, clims=(-0.1, 0.1))
end

mp4(anim, "2d_turbulence_vorticity.mp4", fps = 15) # hide
