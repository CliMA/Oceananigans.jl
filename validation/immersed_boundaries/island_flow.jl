using Printf
using Oceananigans
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom

underlying_grid = RectilinearGrid(CPU(),
                                  size=(128, 64, 16),
                                  halo=(3, 3, 3),
                                  x = (-10, 10), 
                                  y = (-2, 2), 
                                  z = (-1, 1),
                                  topology = (Periodic, Bounded, Bounded))

# Gaussian bump of width "1"
bump(x, y) = -1 + exp(-x^2)
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bump))

Δh = 1
τ = 1
ν₄ = Δh^4 / τ
model = HydrostaticFreeSurfaceModel(grid = grid_with_bump,
                                    momentum_advection = WENO5(),
                                    tracer_advection = WENO5(),
                                    free_surface = ImplicitFreeSurface(),
                                    closure = HorizontalScalarBiharmonicDiffusivity(ν=ν₄, κ=ν₄),
                                    tracers = :b,
                                    buoyancy = BuoyancyTracer(),
                                    coriolis = FPlane(f=sqrt(0.5)))

# Linear stratification
set!(model, b = (x, y, z) -> 4 * z)

progress_message(s) = @info @sprintf("[%.2f%%], iteration: %d, time: %.3f, max|w|: %.2e",
                            100 * s.model.clock.time / s.stop_time, s.model.clock.iteration,
                            s.model.clock.time, maximum(abs, model.velocities.w))

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(10))

simulation = Simulation(model, Δt = 1, stop_iteration=10)

simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                                                    schedule = TimeInterval(0.1),
                                                    prefix = "island_flow",
                                                    force = true)

run!(simulation)

