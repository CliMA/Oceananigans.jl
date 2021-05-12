using Oceananigans, Plots

grid = RegularRectilinearGrid(size=(128, 128), x=(-5, 5), z=(0, 5), topology=(Periodic, Flat, Bounded))

# Gaussian bump
solid(x, y, z) = z < exp(-x^2)

# Tidal forcing
tidal_forcing(x, y, z, t) = 0.001 * cos(t)

model = HydrostaticFreeSurfaceModel(grid = grid,
                                    closure = IsotropicDiffusivity(ν = 1e-4, κ = 1e-4),
                                    tracers = :b,
                                    buoyancy = BuoyancyTracer(),
                                    forcing = (u = tidal_forcing,))
                                    
# Linear stratification with N² = 1
set!(model, b = (x, y, z) -> z)
              
simulation = Simulation(model, Δt = 1e-1, stop_time = 20π)
run!(simulation)

xw, yw, zw = nodes(model.velocities.w)
w = interior(model.velocities.w)[:, 1, :]
pl = contourf(xw, zw, w)

display(pl)
