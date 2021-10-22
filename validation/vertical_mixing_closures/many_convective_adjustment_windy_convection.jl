pushfirst!(LOAD_PATH, joinpath("..", ".."))

using Plots
using Printf
using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: ConvectiveAdjustmentVerticalDiffusivity
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ColumnEnsembleSize

column_ensemble_size = ColumnEnsembleSize(Nz=32, ensemble=(2, 1))

grid = RegularRectilinearGrid(size=column_ensemble_size, z=(-64, 0), topology=(Flat, Flat, Bounded))

closure_1 = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1, background_κz = 1e-5, convective_νz = 1e-3, background_νz = 1e-4)
closure_2 = ConvectiveAdjustmentVerticalDiffusivity(convective_κz = 1, background_κz = 1e-5, convective_νz = 1e-1, background_νz = 1e-4)
                                                  
closures = Matrix{typeof(closure_1)}(undef, 2, 1)
closures[1] = closure_1
closures[2] = closure_2

Qᵇ = +1e-8
Qᵘ = +1e-5

b_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵇ))
u_bcs = FieldBoundaryConditions(top = FluxBoundaryCondition(Qᵘ))

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    tracers = :b,
                                    buoyancy = BuoyancyTracer(),
                                    coriolis = FPlane(f=1e-4),
                                    boundary_conditions = (; b=b_bcs, u=u_bcs),
                                    closure = closures)
                                    
N² = 1e-5
bᵢ(x, y, z) = N² * z
set!(model, b = bᵢ)

z = znodes(model.tracers.b)

b1 = view(interior(model.tracers.b),    1, 1, :)
u1 = view(interior(model.velocities.u), 1, 1, :)
v1 = view(interior(model.velocities.v), 1, 1, :)

b2 = view(interior(model.tracers.b),    2, 1, :)
u2 = view(interior(model.velocities.u), 2, 1, :)
v2 = view(interior(model.velocities.v), 2, 1, :)

b_plot = plot(b1, z, linewidth = 2, label = "t = 0", xlabel = "Buoyancy", ylabel = "z", legend = :bottomright)

u_plot = plot(u1, z, linewidth = 2, label = "u at t = 0",
              linestyle = :solid, color = :black, xlabel = "Velocity", ylabel = "z", legend = :bottomright)

plot!(u_plot, v1, z, linewidth = 2, label = "v at t = 0", linestyle = :dash, color = :black)
              
simulation = Simulation(model, Δt = 20.0, stop_time = 48hours)

run!(simulation)

t = prettytime(model.clock.time)

plot!(b_plot, b1, z, linewidth = 2, label = @sprintf("b₁ at t = %s", t))
plot!(b_plot, b2, z, linewidth = 2, label = @sprintf("b₂ at t = %s", t))

plot!(u_plot, u1, z, label = @sprintf("u₁ at t = %s", t), linewidth = 2, linestyle = :dash,  color = :blue)
plot!(u_plot, v1, z, label = @sprintf("v₁ at t = %s", t), linewidth = 2, linestyle = :solid, color = :blue)

plot!(u_plot, u2, z, label = @sprintf("u₂ at t = %s", t), linewidth = 2, linestyle = :dash,  color = :red)
plot!(u_plot, v2, z, label = @sprintf("v₂ at t = %s", t), linewidth = 2, linestyle = :solid, color = :red)

ub_plot = plot(b_plot, u_plot, layout=(1, 2))

display(ub_plot)
