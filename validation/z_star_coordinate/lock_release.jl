using Oceananigans
using Oceananigans.Units
using Oceananigans.Utils: prettytime
using Oceananigans.Advection: WENOVectorInvariant
using Oceananigans.AbstractOperations: GridMetricOperation  
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ZStar, ZStarSpacingGrid, Δrᶜᶜᶜ
using Printf

grid = RectilinearGrid(size = (20, 20), 
                          y = (0, 64kilometers), 
                          z = (-20, 0), 
                       halo = (6, 6),
                   topology = (Flat, Periodic, Bounded))

grid = ImmersedBoundaryGrid(grid, GridFittedBottom(x -> x < 32kilometers ? -10 : -20))

model = HydrostaticFreeSurfaceModel(; grid, 
            vertical_coordinate = ZStar(),
                         momentum_advection = WENO(; order = 5),
                           tracer_advection = WENO(; order = 5),
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

Δz = GridMetricOperation((Center, Center, Center), Oceananigans.AbstractOperations.Δz, model.grid)

field_outputs = merge(model.velocities, model.tracers, (; Δz))

simulation.output_writers[:other_variables] = JLD2OutputWriter(model, field_outputs, 
                                                               overwrite_existing = true,
                                                               schedule = IterationInterval(100),
                                                               filename = "zstar_model") 

function progress(sim)
    w  = interior(sim.model.velocities.w, :, :, sim.model.grid.Nz+1)
    u  = sim.model.velocities.u
    b  = sim.model.tracers.b
    
    msg0 = @sprintf("Time: %s iteration %d ", prettytime(sim.model.clock.time), sim.model.clock.iteration)
    msg1 = @sprintf("extrema w: %.2e %.2e ",  maximum(w),  minimum(w))
    msg2 = @sprintf("extrema u: %.2e %.2e ",  maximum(u),  minimum(u))
    msg3 = @sprintf("extrema b: %.2e %.2e ",  maximum(b),  minimum(b))
    msg4 = @sprintf("extrema Δz: %.2e %.2e ", maximum(Δz), minimum(Δz))
    @info msg0 * msg1 * msg2 * msg3 * msg4

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

run!(simulation)

using Oceananigans.Utils
using KernelAbstractions: @kernel, @index

@kernel function _compute_field!(tmp, s, b)
  i, j, k = @index(Global, NTuple)
  @inbounds tmp[i, j, k] = s[i, j, k] * b[i, j, k]
end

using Oceananigans.Fields: OneField

# # Check tracer conservation
b  = FieldTimeSeries("zstar_model.jld2", "b")
Δz = FieldTimeSeries("zstar_model.jld2", "Δz")

tmpfield = CenterField(grid)
launch!(CPU(), grid, :xyz, _compute_field!, tmpfield, Δz[1], b[1])
init  = sum(tmpfield) / sum(Δz[1])
drift = []

for t in 1:length(b.times)
  launch!(CPU(), grid, :xyz, _compute_field!, tmpfield, Δz[t], b[t])
  push!(drift, sum(tmpfield) /  sum(Δz[t]) - init) 
end

