# # Ocean convection example
#
# In this example, two-dimensional convection into a stratified fluid
# mixes a phytoplankton-like tracer. This example demonstrates how
#
#   * to set boundary conditions;
#   * to defined and insert a user-defined forcing function into a simulation.
#   * to use the `TimeStepWizard` to manage and adapt the simulation time-step.
#
# To begin, we load Oceananigans, a plotting package, and a few miscellaneous useful packages.

using Random, Printf, Plots
using Oceananigans, Oceananigans.Utils

# ## Parameters
#
# We choose a modest two-dimensional resolution of 128² in a 64² m² domain ,
# implying a resolution of 0.5 m. Our fluid is initially stratified with
# a squared buoyancy frequency
#
# $ N² = 10⁻⁵ \rm{s⁻²} $
#
# and a surface buoyancy flux
#
# $ Q_b = 10⁻⁸ \rm{m³ s⁻²} $
#
# Because we use the physics-based convection whereby buoyancy flux by a
# positive vertical velocity implies positive flux, a positive buoyancy flux
# at the top of the domain carries buoyancy out of the fluid and causes convection.
# Finally, we end the simulation after 1 day.

Nz = 128
Lz = 64.0
N² = 1e-5
Qb = 1e-8
end_time = day / 2
nothing # hide

# ## Creating boundary conditions
#
# Create boundary conditions. Note that temperature is buoyancy in our problem.
#

grid = RegularCartesianGrid(size = (Nz, 1, Nz), length = (Lz, Lz, Lz))

buoyancy_bcs = TracerBoundaryConditions(grid,    top = BoundaryCondition(Flux, Qb),
                                              bottom = BoundaryCondition(Gradient, N²))
nothing # hide

# ## Define a forcing function
#
# Our forcing function roughly corresponds to the growth of phytoplankton in light
# (with a penetration depth of 16 meters here), and death due to natural mortality
# at a rate of 1 phytoplankton unit per second.

growth_and_decay = SimpleForcing((x, y, z, t) -> exp(z/16))

## Instantiate the model
model = Model(
                   grid = grid,
                closure = ConstantIsotropicDiffusivity(ν=1e-4, κ=1e-4),
               coriolis = FPlane(f=1e-4),
                tracers = (:b, :plankton),
               buoyancy = BuoyancyTracer(),
                forcing = ModelForcing(plankton=growth_and_decay),
    boundary_conditions = SolutionBoundaryConditions(grid, b=buoyancy_bcs)
)
nothing # hide

## Set initial condition. Initial velocity and salinity fluctuations needed for AMD.
Ξ(z) = randn() * z / Lz * (1 + z / Lz) # noise
b₀(x, y, z) = N² * z + N² * Lz * 1e-6 * Ξ(z)
set!(model, b=b₀)

## A wizard for managing the simulation time-step.
wizard = TimeStepWizard(cfl=0.1, Δt=1.0, max_change=1.1, max_Δt=90.0)
nothing # hide

# Set up and run the simulation:
simulation = Simulation(model, Δt=wizard, stop_iteration=0, progress_frequency=100)

anim = @animate for i = 1:100
    simulation.stop_iteration += 100
    walltime = @elapsed run!(simulation)

    ## Print a progress message
    @printf("progress: %.1f %%, i: %04d, t: %s, Δt: %s, wall time: %s\n",
            model.clock.time / end_time * 100, model.clock.iteration,
            prettytime(model.clock.time), prettytime(wizard.Δt), prettytime(walltime))

    ## Coordinate arrays for plotting
    xC, zF, zC = model.grid.zC, model.grid.zF[1:Nz], model.grid.zC

    ## Fields to plot (converted to 2D arrays).
    w = Array(interior(model.velocities.w))[:, 1, :]
    p = Array(interior(model.tracers.plankton))[:, 1, :]

    ## Plot the fields.
    w_plot = heatmap(xC, zF, w', xlabel="x (m)", ylabel="z (m)", color=:balance, clims=(-1e-2, 1e-2))
    p_plot = heatmap(xC, zC, p', xlabel="x (m)", ylabel="z (m)", color=:matter) #, legend=false)

    ## Arrange the plots side-by-side.
    plot(w_plot, p_plot, layout=(1, 2), size=(1000, 400),
         title=["vertical velocity (m/s)" "Plankton concentration"])
end

mp4(anim, "ocean_convection_with_plankton.mp4", fps = 15) # hide
