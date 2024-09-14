using Oceananigans
using Oceananigans.Units
using Oceananigans.Utils: prettytime
using Oceananigans.Advection: WENOVectorInvariant
using Printf

grid = RectilinearGrid(size = (128, 20), 
                          x = (0, 64kilometers), 
                          z = (-20, 0), 
                       halo = (6, 6),
                   topology = (Bounded, Flat, Bounded))

model = HydrostaticFreeSurfaceModel(; grid, 
                         momentum_advection = WENO(; order = 5),
                           tracer_advection = WENO(; order = 5),
                                   buoyancy = BuoyancyTracer(),
                                    closure = nothing, 
                                # timestepper = :RungeKutta3,
                                    tracers = :b,
                               free_surface = ImplicitFreeSurface()) #SplitExplicitFreeSurface(; substeps = 40))

g = model.free_surface.gravitational_acceleration

bᵢ(x, z) = x < 32kilometers ? 0.06 : 0.01

set!(model, b = bᵢ)

Δt = 1

@info "the time step is $Δt"

simulation = Simulation(model; Δt, stop_iteration = 100000, stop_time = 17hours) 

field_outputs = merge(model.velocities, model.tracers)

simulation.output_writers[:other_variables] = JLD2OutputWriter(model, field_outputs, 
                                                               overwrite_existing = true,
                                                               schedule = IterationInterval(100),
                                                               filename = "zstar_model") 

function progress(sim)
    w  = interior(sim.model.velocities.w, :, :, sim.model.grid.Nz+1)
    u  = sim.model.velocities.u
    b  = sim.model.tracers.b
    
    msg0 = @sprintf("Time: %s iteration %d ", prettytime(sim.model.clock.time), sim.model.clock.iteration)
    msg1 = @sprintf("extrema w: %.2e %.2e ", maximum(w), minimum(w))
    msg2 = @sprintf("extrema u: %.2e %.2e ", maximum(u), minimum(u))
    msg3 = @sprintf("extrema b: %.2e %.2e ", maximum(b), minimum(b))
    @info msg0 * msg1 * msg2 * msg3
    
    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

run!(simulation)
