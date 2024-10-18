using Oceananigans
using Oceananigans.Units
using Oceananigans.Utils: prettytime
using Oceananigans.Advection: WENOVectorInvariant
using Oceananigans.AbstractOperations: GridMetricOperation  
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ZStarSpacingGrid
using Printf

grid = RectilinearGrid(size = (128, 20), 
                          x = (0, 64kilometers), 
                          z = (-20, 0), 
                       halo = (6, 6),
                   topology = (Bounded, Flat, Bounded))

model = HydrostaticFreeSurfaceModel(; grid, 
                         momentum_advection = WENO(),
                           tracer_advection = Oceananigans.Advection.RotatedAdvection(WENO()),
                                   buoyancy = BuoyancyTracer(),
                                    closure = nothing, 
                                    tracers = :b,
                               free_surface = SplitExplicitFreeSurface(; substeps = 10))

g = model.free_surface.gravitational_acceleration

model.timestepper.χ = 0.0

bᵢ(x, z) = x < 32kilometers ? 0.06 : 0.01

set!(model, b = bᵢ)

Δt = 10

@info "the time step is $Δt"

simulation = Simulation(model; Δt, stop_iteration = 10000, stop_time = 17hours) 

field_outputs = merge(model.velocities, model.tracers)

simulation.output_writers[:other_variables] = JLD2OutputWriter(model, field_outputs, 
                                                               overwrite_existing = true,
                                                               schedule = IterationInterval(100),
                                                               filename = "lock_release") 

function progress(sim)
    w  = interior(sim.model.velocities.w, :, :, sim.model.grid.Nz+1)
    u  = sim.model.velocities.u
    b  = sim.model.tracers.b

    msg0 = @sprintf("Time: %s iteration %d ", prettytime(sim.model.clock.time), sim.model.clock.iteration)
    msg1 = @sprintf("extrema w: %.2e %.2e ",  maximum(w),  minimum(w))
    msg2 = @sprintf("extrema u: %.2e %.2e ",  maximum(u),  minimum(u))
    msg3 = @sprintf("extrema Δz: %.2e %.2e ", maximum(b),  minimum(b))
    @info msg0 * msg1 * msg2 * msg3

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(1))

run!(simulation)
