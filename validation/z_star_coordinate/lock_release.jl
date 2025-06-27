using Oceananigans
using Oceananigans.Grids
using Oceananigans.Units
using Oceananigans.Utils: prettytime
using Oceananigans.Advection: WENOVectorInvariant
using Oceananigans.AbstractOperations: GridMetricOperation
using Printf

z_faces = MutableVerticalDiscretization((-20, 0))

grid = RectilinearGrid(size = (128, 20),
                          x = (0, 64kilometers),
                          z = z_faces,
                       halo = (6, 6),
                   topology = (Bounded, Flat, Bounded))

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(x -> - (64kilometers - x) / 64kilometers * 20))

model = HydrostaticFreeSurfaceModel(; grid,
                         momentum_advection = WENO(order = 5),
                           tracer_advection = WENO(order = 5),
                                   buoyancy = BuoyancyTracer(),
                                    closure = nothing,
                                    tracers = :b,
                                timestepper = :SplitRungeKutta3,
                        vertical_coordinate = Oceananigans.Models.ZStar(),
                               free_surface = SplitExplicitFreeSurface(grid; substeps = 10))

g = model.free_surface.gravitational_acceleration

bᵢ(x, z) = x < 32kilometers ? 0.06 : 0.01

set!(model, b = bᵢ)

Δt = 10

@info "the time step is $Δt"

simulation = Simulation(model; Δt, stop_time = 17hours)

Δz = zspacings(grid, Center(), Center(), Center())
∫b_init = sum(model.tracers.b * Δz) / sum(Δz)

field_outputs = merge(model.velocities, model.tracers, (; Δz))

simulation.output_writers[:other_variables] = JLD2Writer(model, field_outputs,
                                                         overwrite_existing = true,
                                                               schedule = IterationInterval(100),
                                                               filename = "zstar_model")

function progress(sim)
    w  = interior(sim.model.velocities.w, :, :, sim.model.grid.Nz+1)
    u  = sim.model.velocities.u
    ∫b = sum(model.tracers.b * Δz) / sum(Δz)

    msg0 = @sprintf("Time: %s iteration %d ", prettytime(sim.model.clock.time), sim.model.clock.iteration)
    msg1 = @sprintf("extrema w: %.2e %.2e ",  maximum(w),  minimum(w))
    msg2 = @sprintf("extrema u: %.2e %.2e ",  maximum(u),  minimum(u))
    msg3 = @sprintf("drift b: %6.3e ", ∫b - ∫b_init)
    msg4 = @sprintf("extrema Δz: %.2e %.2e ", maximum(Δz), minimum(Δz))
    @info msg0 * msg1 * msg2 * msg3 * msg4

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

run!(simulation)

using Oceananigans.Fields: OneField

# # Check tracer conservation
b  = FieldTimeSeries("zstar_model.jld2", "b")
dz = FieldTimeSeries("zstar_model.jld2", "Δz")

init  = sum(dz[1] * b[1]) / sum(dz[1])
drift = []

for t in 1:length(b.times)
  push!(drift, sum(dz[t] * b[t]) /  sum(dz[t]) - init)
end

using CairoMakie
lines(drift)
