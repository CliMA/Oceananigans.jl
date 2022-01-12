using Plots
using Printf
using Oceananigans
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization, time_discretization

grid = RectilinearGrid(size=128, z=(-0.5, 0.5), topology=(Flat, Flat, Bounded))

evd_closure = HorizontallyCurvilinearAnisotropicDiffusivity(κz = 1.0)
ivd_closure = HorizontallyCurvilinearAnisotropicDiffusivity(κz = 1.0, time_discretization = VerticallyImplicitTimeDiscretization())

model_kwargs = (grid=grid, tracers=:c, buoyancy=nothing, velocities=PrescribedVelocityFields())

implicit_model = HydrostaticFreeSurfaceModel(; closure=ivd_closure, model_kwargs...)
explicit_model = HydrostaticFreeSurfaceModel(; closure=evd_closure, model_kwargs...)
models = (implicit_model, explicit_model)
                                    
initial_temperature(x, y, z) = exp(-z^2 / 0.02)

[set!(model, c=initial_temperature) for model in models]

z = znodes(implicit_model.tracers.c)

c_implicit = view(interior(implicit_model.tracers.c), 1, 1, :)
c_explicit = view(interior(explicit_model.tracers.c), 1, 1, :)

c_plot = plot(c_implicit, z, linewidth = 2, label = "t = 0", xlabel = "Tracer concentration", ylabel = "z")
              
diffusion_time_scale = implicit_model.grid.Δzᵃᵃᶜ^2 / implicit_model.closure.κz.c
stop_time = 100diffusion_time_scale

simulations = [Simulation(explicit_model, Δt = 1e-1 * diffusion_time_scale, stop_time = stop_time),
               Simulation(implicit_model, Δt = 1e0  * diffusion_time_scale, stop_time = stop_time)]

[run!(simulation) for simulation in simulations]

plot!(c_plot, c_implicit, z, linewidth = 3, alpha = 0.6, label = @sprintf("implicit model, t = %.3e", implicit_model.clock.time))
plot!(c_plot, c_explicit, z, linewidth = 2, linestyle = :dash, label = @sprintf("explicit model, t = %.3e", explicit_model.clock.time))

display(c_plot)
