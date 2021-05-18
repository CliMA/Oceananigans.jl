using Plots
using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization, time_discretization, TKEBasedVerticalDiffusivity

grid = RegularRectilinearGrid(size=32, z=(-32, 0), topology=(Flat, Flat, Bounded))

closure = TKEBasedVerticalDiffusivity(time_discretization=VerticallyImplicitTimeDiscretization())

Qᵇ = 1e-7
Qᵉ = - closure.dissipation_constant * closure.surface_model.CᵂwΔ * Qᵇ * grid.Δz

b_bcs = TracerBoundaryConditions(grid; top = FluxBoundaryCondition(Qᵇ))
tke_bcs = TracerBoundaryConditions(grid; top = FluxBoundaryCondition(Qᵉ))

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    tracers = (:b, :e),
                                    buoyancy = BuoyancyTracer(),
                                    boundary_conditions = (b=b_bcs, e=tke_bcs),
                                    closure = closure)
                                    
N² = 1e-5
bᵢ(x, y, z) = N² * z
set!(model, b = bᵢ)

z = znodes(model.tracers.b)

b = view(interior(model.tracers.b), 1, 1, :)
e = view(interior(model.tracers.e), 1, 1, :)

b_plot = plot(b, z, linewidth = 2, label = "t = 0", xlabel = "Buoyancy", ylabel = "z", legend=:bottomright)
e_plot = plot(e, z, linewidth = 2, label = "t = 0", xlabel = "TKE", ylabel = "z", legend=:bottomright)
              
simulation = Simulation(model, Δt = 10.0, stop_iteration = 100)

run!(simulation)

plot!(b_plot, b, z, linewidth = 2, label = @sprintf("t = %.3e", model.clock.time))
plot!(e_plot, e, z, linewidth = 2, label = @sprintf("t = %.3e", model.clock.time))

eb_plot = plot(e_plot, b_plot, layout=(1, 2))

display(eb_plot)
