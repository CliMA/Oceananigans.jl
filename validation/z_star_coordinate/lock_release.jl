using Oceananigans
using Oceananigans.Units
using Oceananigans.Utils: prettytime
using Oceananigans.Advection: WENOVectorInvariant
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ZStar, ZStarSpacingGrid, Δzᶜᶜᶜ_reference
using Printf

grid = RectilinearGrid(size = (128, 20), 
                          x = (0, 64kilometers), 
                          z = (-20, 0), 
                       halo = (6, 6),
                   topology = (Bounded, Flat, Bounded))

model = HydrostaticFreeSurfaceModel(; grid, 
            generalized_vertical_coordinate = ZStar(),
                         momentum_advection = WENOVectorInvariant(),
                           tracer_advection = WENO(),
                                   buoyancy = BuoyancyTracer(),
                                    tracers = :b,
                               free_surface = SplitExplicitFreeSurface(; substeps = 100))

g = model.free_surface.gravitational_acceleration

model.timestepper.χ = 0.0

bᵢ(x, z) = x < 32kilometers ? 0.06 : 0.01

set!(model, b = bᵢ)

Δt = 1

@info "the time step is $Δt"

simulation = Simulation(model; Δt, stop_iteration = 100000, stop_time = 17hours) 

field_outputs = if model.grid isa ZStarSpacingGrid
  merge(model.velocities, model.tracers, (; sⁿ = model.grid.Δzᵃᵃᶠ.sᶜᶜⁿ))
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
    b  = sim.model.tracers.b
    
    msg0 = @sprintf("Time: %s iteration %d ", prettytime(sim.model.clock.time), sim.model.clock.iteration)
    msg1 = @sprintf("extrema w: %.2e %.2e ", maximum(w), minimum(w))
    msg2 = @sprintf("extrema u: %.2e %.2e ", maximum(u), minimum(u))
    msg3 = @sprintf("extrema b: %.2e %.2e ", maximum(b), minimum(b))
    if sim.model.grid isa ZStarSpacingGrid
      Δz = sim.model.grid.Δzᵃᵃᶠ.sᶜᶜⁿ
      msg4 = @sprintf("extrema Δz: %.2e %.2e ", maximum(Δz), minimum(Δz))
      @info msg0 * msg1 * msg2 * msg3 * msg4
    else
      @info msg0 * msg1 * msg2 * msg3
    end

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
b = FieldTimeSeries("zstar_model.jld2", "b")
s = FieldTimeSeries("zstar_model.jld2", "sⁿ")

# s = OneField()
tmpfield = CenterField(grid)
launch!(CPU(), grid, :xyz, _compute_field!, tmpfield, s[1], b[1])
init  = sum(tmpfield) / sum(s[1])
drift = []

for t in 1:length(b.times)
  launch!(CPU(), grid, :xyz, _compute_field!, tmpfield, s[t], b[t])
  push!(drift, sum(tmpfield) /  sum(s[t]) - init) 
end

