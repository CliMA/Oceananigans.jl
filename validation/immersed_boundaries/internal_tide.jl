using Oceananigans, Plots

grid = RegularRectilinearGrid(size=(128, 128), x=(-5, 5), z=(0, 5), topology=(Periodic, Flat, Bounded))

# Gaussian bump
bump(x, y, z) = z < exp(-x^2)

# Tidal forcing
tidal_forcing(x, y, z, t) = 1e-2 * cos(t)

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    momentum_advection = UpwindBiasedThirdOrder(),
                                    closure = IsotropicDiffusivity(ν = 1e-6, κ = 1e-6),
                                    tracers = :b,
                                    buoyancy = BuoyancyTracer(),
                                    immersed_boundary = bump,
                                    forcing = (u = tidal_forcing,))
                                    
# Linear stratification with N² = 1
set!(model, b = (x, y, z) -> z)
              
simulation = Simulation(model, Δt = 2e-3, stop_iteration=10000)
run!(simulation)

xu, yu, zu = nodes(model.velocities.u)
u = interior(model.velocities.u)[:, 1, :]
u_plot = heatmap(xu, zu, u', title="u velocity")

xb, yb, zb = nodes(model.tracers.b)
b = interior(model.tracers.b)[:, 1, :]
b_plot = heatmap(xb, zb, b', title="buoyancy")

ub_plot = plot(u_plot, b_plot, layout=(2, 1))
display(ub_plot)
