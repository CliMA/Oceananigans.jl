using Plots
using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization, TKEBasedVerticalDiffusivity
using Oceananigans.TurbulenceClosures: z_diffusivity, z_viscosity


grid = RegularRectilinearGrid(size=16, z=(-64, 0), topology=(Flat, Flat, Bounded))

closure = TKEBasedVerticalDiffusivity(time_discretization=VerticallyImplicitTimeDiscretization())
                                      
Qᵇ = 1e-8
Qᵘ = - 1e-4
Qᵛ = 0.0

u★ = (Qᵘ^2 + Qᵛ^2)^(1/4)
w★ = Qᵇ * grid.Δz

Qᵉ = - closure.dissipation_parameter * (closure.surface_model.CᵂwΔ * w★ + closure.surface_model.Cᵂu★ * u★)

u_bcs = UVelocityBoundaryConditions(grid; top = FluxBoundaryCondition(Qᵘ))
v_bcs = VVelocityBoundaryConditions(grid; top = FluxBoundaryCondition(Qᵛ))
b_bcs = TracerBoundaryConditions(grid; top = FluxBoundaryCondition(Qᵇ))
tke_bcs = TracerBoundaryConditions(grid; top = FluxBoundaryCondition(Qᵉ))

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    tracers = (:b, :e),
                                    buoyancy = BuoyancyTracer(),
                                    coriolis = FPlane(f=1e-4),
                                    boundary_conditions = (b=b_bcs, e=tke_bcs),
                                    closure = closure)
                                    
N² = 1e-5
bᵢ(x, y, z) = N² * z
set!(model, b = bᵢ)

viscosity_op = z_viscosity(closure, nothing, model.velocities, model.tracers, model.buoyancy)
tracer_diffusivity_op = z_diffusivity(closure, Val(1), nothing, model.velocities, model.tracers, model.buoyancy)
tke_diffusivity_op = z_diffusivity(closure, Val(2), nothing, model.velocities, model.tracers, model.buoyancy)

viscosity = ComputedField(viscosity_op)
tracer_diffusivity = ComputedField(tracer_diffusivity_op)
tke_diffusivity = ComputedField(tke_diffusivity_op)

compute!(viscosity)
compute!(tracer_diffusivity)
compute!(tke_diffusivity)

z = znodes(model.tracers.b)

b = view(interior(model.tracers.b), 1, 1, :)
e = view(interior(model.tracers.e), 1, 1, :)
Ku = view(interior(viscosity), 1, 1, :)
Kc = view(interior(tracer_diffusivity), 1, 1, :)
Ke = view(interior(tke_diffusivity), 1, 1, :)

b_plot = plot(b, z, linewidth = 2, label = "t = 0", xlabel = "Buoyancy", ylabel = "z", legend=:bottomright)
e_plot = plot(e, z, linewidth = 2, label = "t = 0", xlabel = "TKE", ylabel = "z", legend=:bottomright)
              
simulation = Simulation(model, Δt = 1.0, stop_time = 24hours)

run!(simulation)

compute!(viscosity)
compute!(tracer_diffusivity)
compute!(tke_diffusivity)

plot!(b_plot, b, z, linewidth = 2, label = @sprintf("t = %s", prettytime(model.clock.time)))
plot!(e_plot, e, z, linewidth = 2, label = @sprintf("t = %s", prettytime(model.clock.time)))

K_plot = plot(Kc, z, linewidth = 2, linestyle=:dash, label = @sprintf("Kᶜ, t = %s", prettytime(model.clock.time)), legend=:bottomright, xlabel="Diffusivities")
plot!(K_plot, Ke, z, linewidth = 3, alpha=0.6, label = @sprintf("Kᵉ, t = %s", prettytime(model.clock.time)))
plot!(K_plot, Ku, z, linewidth = 2, linestyle=:dotted, label = @sprintf("Kᵘ, t = %s", prettytime(model.clock.time)))

eb_plot = plot(b_plot, e_plot, K_plot, layout=(1, 3))

display(eb_plot)
