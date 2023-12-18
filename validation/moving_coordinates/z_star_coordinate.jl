using Oceananigans
using Oceananigans.Units
using Oceananigans.Utils: prettytime
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ZStarCoordinate, ZCoordinate, ZStarCoordinateGrid
using Printf

grid = RectilinearGrid(size = (300, 20), 
                          x = (0, 100kilometers), 
                          z = (-10, 0), 
                   topology = (Periodic, Flat, Bounded))

model = HydrostaticFreeSurfaceModel(; grid, 
                        vertical_coordinate = ZStarCoordinate(),
                         momentum_advection = WENO(),
                           tracer_advection = WENO(),
                                   buoyancy = BuoyancyTracer(),
                                    tracers = :b,
                               free_surface = SplitExplicitFreeSurface(; cfl = 0.5, grid))

ηᵢ(x, z) = exp(-(x - 50kilometers)^2 / (10kilometers)^2)
bᵢ(x, z) = 1e-6 * z + ηᵢ(x, z) * 1e-8

set!(model, η = ηᵢ, b = bᵢ)

gravity_wave_speed   = sqrt(model.free_surface.gravitational_acceleration * grid.Lz)
barotropic_time_step = grid.Δxᶜᵃᵃ / gravity_wave_speed

Δt = 0.5 * barotropic_time_step

@info "the time step is $Δt"

simulation = Simulation(model; Δt, stop_time = 10000Δt)

field_outputs = if model.grid isa ZStarCoordinateGrid
  merge(model.velocities, model.tracers, (; ΔzF = model.grid.Δzᵃᵃᶠ.star_value))
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
    
    msg0 = @sprintf("Time: %s iteration %d ", prettytime(sim.model.clock.time), sim.model.clock.iteration)
    msg1 = @sprintf("extrema w: %.2e %.2e ", maximum(w), minimum(w))
    msg2 = @sprintf("extrema u: %.2e %.2e ", maximum(u), minimum(u))
    if sim.model.grid isa ZStarCoordinateGrid
      Δz = sim.model.grid.Δzᵃᵃᶠ.star_value
      msg3 = @sprintf("extrema Δz: %.2e %.2e ", maximum(Δz), minimum(Δz))
      @info msg0 * msg1 * msg2 * msg3
    else
      @info msg0 * msg1 * msg2
    end

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(1))

run!(simulation)

# Check conservation
b = FieldTimeSeries("zstar_model.jld2", "b")

drift = []
for t in 1:length(b.times)
  push!(drift, sum(b[t] - b[1]))
end