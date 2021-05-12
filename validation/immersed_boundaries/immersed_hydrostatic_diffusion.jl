using Plots
using Printf
using Oceananigans
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization, time_discretization

grid = RegularRectilinearGrid(size=128, z=(-0.5, 0.5), topology=(Flat, Flat, Bounded))

closure = HorizontallyCurvilinearAnisotropicDiffusivity(κz = 1.0)

model_kwargs = (tracers=:c, buoyancy=nothing, velocities=PrescribedVelocityFields())

model = HydrostaticFreeSurfaceModel(; grid=grid, closure=closure, model_kwargs...)
                                    
initial_temperature(x, y, z) = exp(-z^2 / 0.02)
set!(model, c=initial_temperature)
z = znodes(model.tracers.c)

c = view(interior(model.tracers.c), 1, 1, :)

c_plot = plot(c, z, linewidth = 2, label = "t = 0", xlabel = "Tracer concentration", ylabel = "z")
              
diffusion_time_scale = implicit_model.grid.Δz^2 / implicit_model.closure.κz.c
stop_time = 100diffusion_time_scale

simulation = Simulation(explicit_model, Δt = 1e-1 * diffusion_time_scale, stop_time = stop_time)
run!(simulation)

plot!(c_plot, c, z, linewidth = 3, alpha = 0.6, label = @sprintf("NoImmersedBoundary model, t = %.3e", model.clock.time))

display(c_plot)
