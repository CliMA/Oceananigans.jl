# # Two dimensional turbulence example
#
# In this example, we initialize a random velocity field and observe its viscous,
# turbulent decay in a two-dimensional domain. This example demonstrates:
#
#   * How to run a model with no buoyancy equation or tracers;
#   * How to create user-defined fields
#   * How to use differentiation functions
#
# For this example, we need `PyPlot` for plotting and `Statistics` for setting up
# a random initial condition with zero mean velocity.

using Oceananigans, PyPlot, Statistics

# In addition to importing plotting and statistics packages, we import
# some types and functions from `Oceananigans` that will aid in the calculation
# and visualization of voriticty.

using Oceananigans: Face, Cell
using Oceananigans.TurbulenceClosures: ∂x_faa, ∂y_afa

# We instantiate the model with a simple isotropic diffusivity

model = Model(
        grid = RegularCartesianGrid(size=(128, 128, 1), grid=(2π, 2π, 2π)),
    buoyancy = nothing,
     tracers = nothing,
     closure = ConstantIsotropicDiffusivity(ν=1e-3, κ=1e-3)
)

# Our initial condition randomizes `u` and `v`. We also ensure that both have
# zero mean for purely aesthetic reasons.

u₀ = rand(size(model.grid)...)
u₀ .-= mean(u₀)

set!(model, u=u₀, v=u₀)

# Next we define a function for calculating the vertical vorticity
# associated with the velocity fields `u` and `v`.

function vorticity!(ω, u, v)
    for j = 1:u.grid.Ny, i = 1:u.grid.Nx
        @inbounds ω.data[i, j, 1] = ∂x_faa(i, j, 1, u.grid, v.data) - ∂y_afa(i, j, 1, u.grid, u.data)
    end
    return nothing
end

# Finally, we create the vorticity field for storing `u` and `v`, initialize a
# figure, and run the model forward

ω = Field(Face, Face, Cell, model.architecture, model.grid)

close("all")
fig, ax = subplots()

for i = 1:10
    time_step!(model, Nt=100, Δt=1e-1)

    vorticity!(ω, model.velocities.u, model.velocities.v)

    cla()
    imshow(data(ω)[:, :, 1])
    ax.axis("off")
    pause(0.1)
end

# We can plot out the final vorticity field.
gcf()
