using Oceananigans
using Oceananigans.Grids
using Oceananigans.Units
using Oceananigans.Utils: prettytime
using Oceananigans.Advection: WENOVectorInvariant
using Oceananigans.AbstractOperations: GridMetricOperation
using Printf

z_faces = MutableVerticalDiscretization((-500, 0))

grid = RectilinearGrid(size = (128, 5),
                          x = (0, 64kilometers),
                          z = z_faces,
                       halo = (6, 6),
                   topology = (Periodic, Flat, Bounded))

# grid = ImmersedBoundaryGrid(grid, GridFittedBottom(x -> - (64kilometers - x) / 64kilometers * 20))

model = HydrostaticFreeSurfaceModel(; grid,
                         momentum_advection = WENO(order = 5),
                           tracer_advection = WENO(order = 5),
                                   buoyancy = BuoyancyTracer(),
                                    closure = nothing,
                                    tracers = (:b, :c),
                                # timestepper = :SplitRungeKutta3,
                        vertical_coordinate = ZStarCoordinate(grid),
                               free_surface = SplitExplicitFreeSurface(grid; substeps=10)) # 

g = model.free_surface.gravitational_acceleration
bᵢ(x, z) = x < 32kilometers ? 0.06 : 0.01

set!(model, b = bᵢ, c = 1)

gravity_wave_speed = sqrt(g * grid.Lz) # hydrostatic (shallow water) gravity wave speed
wave_propagation_time_scale = model.grid.Δxᶜᵃᵃ / gravity_wave_speed

@show Δt = 1 * wave_propagation_time_scale

@info "the time step is $Δt"

simulation = Simulation(model; Δt, stop_time = 20hours) #, stop_iteration=1000000) 

Δz = zspacings(grid, Center(), Center(), Center())
dz = Field(Δz)
V  = KernelFunctionOperation{Center, Center, Center}(Oceananigans.Operators.Vᶜᶜᶜ, grid)
∫b_init = sum(model.tracers.b * Δz) / sum(Δz)

field_outputs = merge(model.velocities, model.tracers, (; Δz))

simulation.output_writers[:other_variables] = JLD2Writer(model, field_outputs,
                                                         overwrite_existing = true,
                                                         schedule = IterationInterval(1000),
                                                         filename = "zstar_model")

et1 = []
et2 = []
bav = []
cav = []
vav = []

function progress(sim)
    w  = interior(sim.model.velocities.w, :, :, sim.model.grid.Nz+1)
    u  = sim.model.velocities.u
    compute!(dz)
    ∫b = sum(model.tracers.b * dz) / sum(dz)
    push!(bav, ∫b)
    push!(cav, sum(model.tracers.c * dz) / sum(dz))
    push!(vav, sum(V))

    msg0 = @sprintf("Time: %s iteration %d ", prettytime(sim.model.clock.time), sim.model.clock.iteration)
    msg1 = @sprintf("extrema w: %.2e %.2e ",  maximum(w),  minimum(w))
    msg2 = @sprintf("extrema u: %.2e %.2e ",  maximum(u),  minimum(u))
    msg3 = @sprintf("drift b: %6.3e ", ∫b - ∫b_init)
    msg4 = @sprintf("extrema Δz: %.2e %.2e ", maximum(Δz), minimum(Δz))
    @info msg0 * msg1 * msg2 * msg3 * msg4

    push!(et1, deepcopy(interior(model.free_surface.η, :, 1, 1)))
    push!(et2, deepcopy(model.grid.z.ηⁿ[1:128, 1, 1]))

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(1000))

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

using GLMakie
GLMakie.activate!()
lines(drift)
