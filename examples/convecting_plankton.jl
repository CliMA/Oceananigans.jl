# # Convecting plankton
#
# In this example, two-dimensional convection into a stratified fluid
# mixes a phytoplankton-like tracer. This example demonstrates:
#
#   * How to set boundary conditions.
#   * How to insert a user-defined forcing function into a model.
#   * How to use the `TimeStepWizard` to adapt the simulation time-step.
#   * How to use `AveragedField` to diagnose spatial averages of model fields.
#
# ## The grid
#
# We use a two-dimensional grid with 128² points and 2 m grid spacing:

using Oceananigans

grid = RegularCartesianGrid(size=(128, 1, 128), extent=(64, 1, 64))

# ## Boundary conditions
#
# We impose buoyancy loss at the surface with the buoyancy flux

Qb = 1e-8 # m³ s⁻²

# Note that a _positive_ flux at the _top_ boundary means that buoyancy is
# carried _upwards_, out of the fluid. This reduces the fluid's buoyancy 
# near the surface, causing convection.
#
# The initial condition consists of the constant buoyancy gradient

N² = 1e-6 # s⁻²

# which we also impose as a boundary condition at the bottom.

buoyancy_bcs = TracerBoundaryConditions(grid, top = BoundaryCondition(Flux, Qb),
                                              bottom = BoundaryCondition(Gradient, N²))

# ## Forcing: the growing and grazing of phytoplankton
#
# We add a forcing term to the plankton equation that crudely models:
#
#   * the growth of phytoplankton at ``γ = 10`` "phytoplankton units" per day,
#     in sunlight with a penetration depth of ``λ = 16`` meters;
#
#   * death due to viruses and grazing by zooplankton at a rate of
#     ``μ = 1`` phytoplankton unit per day.

using Oceananigans.Utils: day

growing_and_grazing(x, y, z, t, p) = p.γ * exp(z / p.λ) - p.μ
    
plankton_forcing = Forcing(growing_and_grazing, parameters=(λ=16, γ=10/day, μ=1/day))

# Finally, we're ready to build an `IncompressibleModel`. We use a third-order 
# advection scheme, third-order Runge-Kutta time-stepping, and add Coriolis
# forces appropriate for planktonic convection at mid-latitudes on Earth.

using Oceananigans.Advection: UpwindBiasedThirdOrder

model = IncompressibleModel(
                   grid = grid,
              advection = UpwindBiasedThirdOrder(),
            timestepper = :RungeKutta3,
                closure = IsotropicDiffusivity(ν=1e-4, κ=1e-4),
               coriolis = FPlane(f=1e-4),
                tracers = (:b, :plankton),
               buoyancy = BuoyancyTracer(),
                forcing = (plankton=plankton_forcing,),
    boundary_conditions = (b=buoyancy_bcs,)
)

# ## Initial condition
#
# Our initial condition consists of a linear buoyancy gradient superposed with
# random noise.

Ξ(z) = randn() * z / grid.Lz * (1 + z / grid.Lz) # noise

initial_buoyancy(x, y, z) = N² * z + N² * grid.Lz * 1e-6 * Ξ(z)

set!(model, b=initial_buoyancy)

# ## Simulation setup
#
# We use a `TimeStepWizard` that limits the
# time-step to 1 minute, and adapts the time-step such that CFL
# (Courant-Freidrichs-Lewy) number remains below 1.0,

using Oceananigans.Utils: minute

wizard = TimeStepWizard(cfl=1.0, Δt=2minutes, max_change=1.1, max_Δt=2minutes)

# We also write a function that prints the progress of the simulation

using Printf
using Oceananigans.Utils: hour, prettytime

progress(sim) = @printf("Iteration: %d, time: %s, Δt: %s\n",
                        sim.model.clock.iteration,
                        prettytime(sim.model.clock.time),
                        prettytime(sim.Δt.Δt))
                               
simulation = Simulation(model, Δt=wizard, stop_time=4hour,
                        iteration_interval=10, progress=progress)

# We add a basic `JLD2OutputWriter` that writes velocities, tracers,
# and the horizontally-averaged plankton:

using Oceananigans.OutputWriters, Oceananigans.Fields

averaged_plankton = AveragedField(model.tracers.plankton, dims=(1, 2))

outputs = (w = model.velocities.w,
           plankton = model.tracers.plankton,
           averaged_plankton = averaged_plankton)

simulation.output_writers[:fields] =
    JLD2OutputWriter(model, outputs,
                     schedule = TimeInterval(10minute),
                     prefix = "convecting_plankton",
                     force = true)

# Note that it often makes sense to define different output writers
# for two- or three-dimensional fields and `AveragedField`s (since 
# averages take up so much less disk space, it's usually possible to output
# them a lot more frequently than full fields without blowing up your hard drive).
#
# The simulation is setup. Let there be plankton:

run!(simulation)

# ## Visualizing the solution
#
# We'd like to a make a plankton movie. First we load the output file,

using JLD2

file = jldopen(simulation.output_writers[:fields].filepath)

iterations = parse.(Int, keys(file["timeseries/t"]))

# Next we construct the ``x, z`` grid for plotting purposes,

using Oceananigans.Grids

xw, yw, zw = nodes(model.velocities.w)
xp, yp, zp = nodes(model.tracers.plankton)

# Finally, we animate the convective plumes and plankton swirls,

using Plots

@info "Making a movie about plankton..."

anim = @animate for (i, iteration) in enumerate(iterations)

    @info "Plotting frame $i from iteration $iteration..."
    
    t = file["timeseries/t/$iteration"]
    w = file["timeseries/w/$iteration"][:, 1, :]
    p = file["timeseries/plankton/$iteration"][:, 1, :]
    P = file["timeseries/averaged_plankton/$iteration"][1, 1, :]

    w_max = maximum(abs, w) + 1e-9
    w_lim = 0.8 * w_max

    p_min = minimum(p) - 1e-9
    p_max = maximum(p) + 1e-9
    p_lim = 1

    w_levels = vcat([-w_max], range(-w_lim, stop=w_lim, length=21), [w_max])
    p_levels = collect(range(p_min, stop=p_lim, length=20))
    p_max > p_lim && push!(p_levels, p_max)

    kwargs = (xlabel="x", ylabel="y", aspectratio=1, linewidth=0, colorbar=true,
              xlims=(0, model.grid.Lx), ylims=(-model.grid.Lz, 0))

    w_plot = contourf(xw, zw, w';
                       color = :balance,
                      levels = w_levels,
                       clims = (-w_lim, w_lim),
                      kwargs...)

    p_plot = contourf(xp, zp, p';
                       color = :matter,
                      levels = p_levels,
                       clims = (p_min, p_lim),
                      kwargs...)

    P_plot = plot(P, zp[:],
                  linewidth = 2,
                  label = nothing,
                  xlims = (-0.2, 1),
                  ylabel = "Averaged plankton",
                  xlabel = "Plankton concentration")

    plot(w_plot, p_plot, P_plot,
         title=["Vertical velocity" "Plankton" "Averaged plankton"],
         link = :y,
         layout=(1, 3), size=(2000, 400))
end

mp4(anim, "convecting_plankton.mp4", fps = 8) # hide
