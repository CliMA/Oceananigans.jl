using Plots
using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: ConvectiveAdjustmentVerticalDiffusivity

grid = RegularRectilinearGrid(size=32, z=(-64, 0), topology=(Flat, Flat, Bounded))

closure = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1,
                                                  background_κz = 1e-5,
                                                  convective_νz = 1e-3,
                                                  background_νz = 1e-4)
                                      
Qᵇ = +1e-8
Qᵘ = +1e-5

b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ))
u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ))

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    tracers = :b,
                                    buoyancy = BuoyancyTracer(),
                                    coriolis = FPlane(f=1e-4),
                                    boundary_conditions = (; b=b_bcs, u=u_bcs),
                                    closure = closure)
                                    
N² = 1e-5
bᵢ(x, y, z) = N² * z
set!(model, b = bᵢ)

z = znodes(model.tracers.b)

b = view(interior(model.tracers.b), 1, 1, :)
u = view(interior(model.velocities.u), 1, 1, :)
v = view(interior(model.velocities.v), 1, 1, :)

b_plot = plot(b, z, linewidth = 2, label = "t = 0", xlabel = "Buoyancy", ylabel = "z", legend=:bottomright)

u_plot = plot(u, z, linewidth = 2, label = "u at t = 0",
              linestyle = :solid, color = :black, xlabel = "velocity", ylabel = "z", legend = :bottomright)

plot!(u_plot, v, z, linewidth = 2, label = "v at t = 0",
      linestyle = :dash, color = :black)
              
simulation = Simulation(model, Δt = 20.0, stop_time = 48hours)

run!(simulation)

plot!(b_plot, b, z, linewidth = 2, label = @sprintf("t = %s", prettytime(model.clock.time)))

plot!(u_plot, u, z, label = @sprintf("u at t = %s", prettytime(model.clock.time)),
      linewidth = 2, linestyle = :dash, color = :blue)

plot!(u_plot, v, z, label = @sprintf("v at t = %s", prettytime(model.clock.time)),
      linewidth = 2, linestyle = :solid, color = :blue)

ub_plot = plot(b_plot, u_plot, layout=(1, 2))

display(ub_plot)
