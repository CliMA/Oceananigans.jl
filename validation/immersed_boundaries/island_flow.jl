using Printf
using Oceananigans
using Oceananigans.AdvectionDivergence: VelocityStencil
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

Δh = underlying_grid.Δxᶜᵃᵃ
τ = 10
ν₄ = Δh^4 / τ
model = HydrostaticFreeSurfaceModel(; grid,
                                    momentum_advection = WENO5(vector_invariant=VelocityStencil()),
                                    tracer_advection = WENO5(),
                                    free_surface = ImplicitFreeSurface(),
                                    closure = HorizontalScalarBiharmonicDiffusivity(ν=ν₄, κ=ν₄),
                                    tracers = :b,
                                    buoyancy = BuoyancyTracer(),
                                    coriolis = FPlane(f=sqrt(0.5)))

N² = 4
S² = 1
bᵢ(x, y, z) = 4 * z
uᵢ(x, y, z) = sqrt(S²) * z + 1
set!(model, b=bᵢ, u=uᵢ)

Δt = 0.1 * Δh # CFL=0.1 with max(u) = 1
simulation = Simulation(model; Δt, stop_iteration=10)

progress_message(s) = @info @sprintf("Iter: %d, t: %.3f, Δt: %.3f, max|w|: %.2e",
                                     iteration(s), time(s), s.Δt, maximum(abs, model.velocities.w))

simulation.callbacks[:progress] = Callback(progress_message, IterationInterval(10))

wizard = TimeStepWizard(cfl=0.1)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                                                    schedule = TimeInterval(0.1),
                                                    prefix = "island_flow",
                                                    force = true)

run!(simulation)

