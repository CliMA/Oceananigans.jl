using Oceananigans
using Oceananigans.Units
using Oceananigans.Utils: prettytime
using Oceananigans.Advection: WENOVectorInvariant
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ZStar, ZStarSpacingGrid, Δzᶜᶜᶜ_reference
using Printf

grid = RectilinearGrid(size = (10, 10), 
                          x = (0, 100kilometers), 
                          z = (-10, 0), 
                       halo = (6, 6),
                   topology = (Bounded, Flat, Bounded))

model = HydrostaticFreeSurfaceModel(; grid, 
            generalized_vertical_coordinate = nothing,
                         momentum_advection = Centered(),
                           tracer_advection = Centered(),
                                   buoyancy = BuoyancyTracer(),
                                    tracers = :b,
                               free_surface = SplitExplicitFreeSurface(; substeps = 10))

g = model.free_surface.gravitational_acceleration

bᵢ(x, z) = x < 50kilometers ? 1 : 0

set!(model, b = bᵢ)

gravity_wave_speed   = sqrt(g * grid.Lz)
barotropic_time_step = grid.Δxᶜᵃᵃ / gravity_wave_speed

Δt = 0.5 * barotropic_time_step

@info "the time step is $Δt"

simulation = Simulation(model; Δt, stop_iteration = 100000, stop_time = 10days) 

field_outputs = if model.grid isa ZStarSpacingGrid
  merge(model.velocities, model.tracers, (; sⁿ = model.grid.Δzᵃᵃᶠ.sⁿ))
else
  merge(model.velocities, model.tracers)
end

simulation.output_writers[:other_variables] = JLD2OutputWriter(model, field_outputs, 
                                                               overwrite_existing = true,
                                                               schedule = IterationInterval(100),
                                                               filename = "zstar_model") 

function progress(sim)
    w  = interior(sim.model.velocities.w, :, :, sim.model.grid.Nz+1)
    u  = sim.model.velocities.u
    v  = sim.model.velocities.v
    η  = sim.model.free_surface.η
    b  = sim.model.tracers.b
    
    msg0 = @sprintf("Time: %s iteration %d ", prettytime(sim.model.clock.time), sim.model.clock.iteration)
    msg1 = @sprintf("extrema w: %.2e %.2e ", maximum(w), minimum(w))
    msg2 = @sprintf("extrema u: %.2e %.2e ", maximum(u), minimum(u))
    msg3 = @sprintf("extrema b: %.2e %.2e ", maximum(b), minimum(b))
    if sim.model.grid isa ZStarSpacingGrid
      Δz = sim.model.grid.Δzᵃᵃᶠ.sⁿ
      msg4 = @sprintf("extrema Δz: %.2e %.2e ", maximum(Δz), minimum(Δz))
      @info msg0 * msg1 * msg2 * msg3 * msg4
    else
      @info msg0 * msg1 * msg2 * msg3
    end

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))
simulation.callbacks[:wizard]   = Callback(TimeStepWizard(; cfl = 0.2, max_change = 1.1), IterationInterval(10))

run!(simulation)

using Oceananigans.Utils

@kernel function _compute_field!(tmp, s, b)
  i, j, k = @index(Global, NTuple)
  @inbounds tmp[i, j, k] = s[i, j, k] * b[i, j, k]
end

using Oceananigans.Fields: OneField

# # Check tracer conservation
b = FieldTimeSeries("zstar_model.jld2", "b")
# s = FieldTimeSeries("zstar_model.jld2", "sⁿ")
s = OneField()
tmpfield = CenterField(grid)
launch!(CPU(), grid, :xyz, _compute_field!, tmpfield, s, b[1])
init  = sum(tmpfield) / 10 # sum(s[1])
drift = []
for t in 1:length(b.times)
  launch!(CPU(), grid, :xyz, _compute_field!, tmpfield, s, b[t])
  push!(drift, sum(tmpfield) / 10 - init) 
end

