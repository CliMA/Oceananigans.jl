using Oceananigans
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary
using Printf
using Plots

solid(x, y, z) = z < 0

underlying_grid = RectilinearGrid(size=128, z=(-0.5, 0.5), topology=(Flat, Flat, Bounded))
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBoundary(solid))

closure = IsotropicDiffusivity(κ = 1.0)

model_kwargs = (tracers=:c, buoyancy=nothing, velocities=PrescribedVelocityFields())

model = HydrostaticFreeSurfaceModel(; grid=underlying_grid, closure=closure, model_kwargs...)
immersed_model = HydrostaticFreeSurfaceModel(; grid=grid, closure=closure, model_kwargs...)
                                    
initial_temperature(x, y, z) = exp(-z^2 / 0.02)
[set!(m, c=initial_temperature) for m in (model, immersed_model)]

z = znodes(model.tracers.c)
c = view(interior(model.tracers.c), 1, 1, :)
c_immersed = view(interior(immersed_model.tracers.c), 1, 1, :)
c_plot = plot(c, z, linewidth = 2, label = "t = 0", xlabel = "Tracer concentration", ylabel = "z")
plot!(c_plot, c_immersed, z, linewidth = 2, label = "t = 0, immersed model", xlabel = "Tracer concentration", ylabel = "z")
              
diffusion_time_scale = model.grid.Δz^2 / model.closure.κ.c
stop_time = 100diffusion_time_scale

simulations = [simulation = Simulation(m, Δt = 1e-1 * diffusion_time_scale, stop_time = stop_time) for m in (model, immersed_model)]
[run!(sim) for sim in simulations]

plot!(c_plot, c, z, linewidth = 2, alpha = 0.6, linestyle = :dash, label = @sprintf("Ordinary model, t = %.3e", model.clock.time))
plot!(c_plot, c_immersed, z, linewidth = 3, alpha = 0.6, label = @sprintf("Immersed model, t = %.3e", immersed_model.clock.time))

display(c_plot)
