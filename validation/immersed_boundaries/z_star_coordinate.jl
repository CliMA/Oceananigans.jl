using Oceananigans
using Oceananigans.Units
using Printf
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ZStarCoordinate

grid = RectilinearGrid(size = (600, 20), x = (0, 18kilometers), z = (-200, 0), topology = (Periodic, Flat, Bounded), halo = (5, 5))

bottom(x) = - 200 + exp( - (x - 9kilometers)^2 / (1.2kilometers)^2) * 150

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(bottom))

@inline u_bc(i, j, grid, clock, fields, p) = 
     @inbounds - 1 / p.λ * (0.1 - fields.u[i, j, grid.Nz])
    

u_boundary = FluxBoundaryCondition(u_bc, discrete_form = true, parameters = (; λ = 10hour))

u_bcs = FieldBoundaryConditions(top = u_boundary)

model = HydrostaticFreeSurfaceModel(; grid, 
                                      free_surface = SplitExplicitFreeSurface(; cfl = 0.7, grid), 
                                      momentum_advection = WENO(), 
                                      tracer_advection = WENO(; order = 7),
                                      tracers = :b,
                                      boundary_conditions = (; u = u_bcs),
                                      closure = VerticalScalarDiffusivity(ν = 1e-3),
                                      buoyancy = BuoyancyTracer(),
                                      vertical_coordinate = ZStarCoordinate())

N² = 0.01
bᵢ(x, z) = N² * z

set!(model, b = bᵢ)

simulation = Simulation(model, Δt = 1, stop_time = 2days, stop_iteration = 10000)

function progress(sim)
    u, v, w = sim.model.velocities 
    @info @sprintf("Time: %s, Δt: %.2f, iteration: %d, velocities: (%.2f, %.2f)", prettytime(sim.model.clock.time), 
                    sim.Δt, sim.model.clock.iteration, maximum(abs, u), maximum(abs, w))
end

wizard = TimeStepWizard(; cfl = 0.2, max_change = 1.1)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(2))
simulation.callbacks[:progress] = Callback(progress, IterationInterval(1))

simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers), 
                                                      filename = "z_star_coordinate",
                                                      overwrite_existing = true,
                                                      schedule = TimeInterval(0.5hour)) 

run!(simulation)