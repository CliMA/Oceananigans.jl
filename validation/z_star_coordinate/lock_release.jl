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

model = HydrostaticFreeSurfaceModel(; grid,
                         momentum_advection = WENO(order=5),
                           tracer_advection = WENO(order=7),
                                   buoyancy = BuoyancyTracer(),
                                    closure = (VerticalScalarDiffusivity(ν=1e-4), HorizontalScalarDiffusivity(ν=1.0)),
                                    tracers = (:b, :c),
                                timestepper = :SplitRungeKutta3,
                        vertical_coordinate = ZStarCoordinate(grid),
                               free_surface = SplitExplicitFreeSurface(grid; substeps=20)) # 

g = model.free_surface.gravitational_acceleration
bᵢ(x, z) = x > 32kilometers ? 0.06 : 0.01

set!(model, b = bᵢ, c = 1)

# Same timestep as in the ilicak paper
Δt = 1

@info "the time step is $Δt"

simulation = Simulation(model; Δt, stop_time=17hours)

Δz = zspacings(grid, Center(), Center(), Center())
V  = KernelFunctionOperation{Center, Center, Center}(Oceananigans.Operators.Vᶜᶜᶜ, grid)

field_outputs = merge(model.velocities, model.tracers, (; Δz))

simulation.output_writers[:other_variables] = JLD2Writer(model, field_outputs,
                                                         overwrite_existing = true,
                                                         schedule = IterationInterval(100),
                                                         filename = "zstar_model")

# The two different estimates of the free-surface
et1 = []
et2 = []

# Initial conditions
bav = [sum(model.tracers.b * V) / sum(V)]
cav = [sum(model.tracers.c * V) / sum(V)]
vav = [sum(V)]

function progress(sim)
    w  = interior(sim.model.velocities.w, :, :, sim.model.grid.Nz+1)
    u  = sim.model.velocities.u
    push!(bav, sum(model.tracers.b * V) / sum(V))
    push!(cav, sum(model.tracers.c * V) / sum(V))
    push!(vav, sum(V))

    msg0 = @sprintf("Time: %s iteration %d ", prettytime(sim.model.clock.time), sim.model.clock.iteration)
    msg1 = @sprintf("extrema w: %.2e %.2e ",  maximum(w),  minimum(w))
    msg2 = @sprintf("extrema u: %.2e %.2e ",  maximum(u),  minimum(u))
    msg3 = @sprintf("drift b: %6.3e ", bav[end] - bav[1])
    msg4 = @sprintf("extrema Δz: %.2e %.2e ", maximum(Δz), minimum(Δz))
    @info msg0 * msg1 * msg2 * msg3 * msg4

    push!(et1, deepcopy(interior(model.free_surface.η, :, 1, 1)))
    push!(et2, deepcopy(model.grid.z.ηⁿ[1:128, 1, 1]))

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

run!(simulation)
 
b = model.tracers.b
x, y, z = nodes(b)

fig = Figure()
ax  = Axis(fig[1, 1], title = "Integral property conservation")
lines!(ax, (vav .- vav[1]) ./ vav[1], label = "Volume anomaly")
lines!(ax, (bav .- bav[1]) ./ bav[1], label = "Buoyancy anomaly")
lines!(ax, (cav .- cav[1]) ./ cav[1], label = "Tracer anomaly")
axislegend(ax, position=:lt)
ax  = Axis(fig[1, 2], title = "Final buoyancy field") 
contourf!(ax, x, z, interior(model.tracers.b, :, 1, :), colormap=:balance, levels=20)
vlines!(ax, 62.3e3, linestyle = :dash, linewidth = 3, color = :black)
