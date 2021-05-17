using Plots
using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization, time_discretization, TKEBasedVerticalDiffusivity

grid = RegularRectilinearGrid(size=128, z=(-128, 0), topology=(Flat, Flat, Bounded))

closure = TKEBasedVerticalDiffusivity()

Qᵇ = 1e-7
Qᵉ = - closure.dissipation_constant * closure.surface_model.CᵂwΔ * Qᵇ * grid.Δz

convection_bcs = TracerBoundaryConditions(grid; top = FluxBoundaryCondition(Qᵇ))
tke_bcs = TracerBoundaryConditions(grid; top = FluxBoundaryCondition(Qᵉ))

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    tracers = (:b, :e),
                                    buoyancy = BuoyancyTracer(),
                                    boundary_conditions = (b = convection_bcs,),
                                    closure = closure)
                                    
N² = 1e-5
bᵢ(x, y, z) = N² * z
set!(model, b = bᵢ)

z = znodes(model.tracers.b)

b = view(interior(model.tracers.b), 1, 1, :)

b_plot = plot(b, z, linewidth = 2, label = "t = 0", xlabel = "Buoyancy", ylabel = "z", legend=:bottomright)
              
simulation = Simulation(model, Δt = 1.0, stop_time = 1hour)

run!(simulation)

plot!(b_plot, b, z, linewidth = 2, label = @sprintf("t = %.3e", model.clock.time))

display(b_plot)
