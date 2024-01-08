using Oceananigans
using Oceananigans.Units
using Oceananigans.Utils: prettytime
using Oceananigans.Operators
using Oceananigans.Advection: WENOVectorInvariant
using Oceananigans.Fields: ZeroField
using Oceananigans.Grids: architecture
using Oceananigans.Models.HydrostaticFreeSurfaceModels: ZStar, ZStarSpacingGrid, update_state!
using Printf

grid = RectilinearGrid(size = (300, 20), 
                          x = (0, 100kilometers), 
                          z = (-10, 0), 
                       halo = (6, 6),
                   topology = (Bounded, Flat, Bounded))

model = HydrostaticFreeSurfaceModel(; grid, 
           generalized_vertical_coordinate = ZStar(),
                        momentum_advection = nothing,
                          tracer_advection = WENO(),
                                   closure = nothing,
                                  buoyancy = nothing,
                                   tracers = (),
                              free_surface = SplitExplicitFreeSurface(; substeps = 30))

g = model.free_surface.gravitational_acceleration

ηᵢ(x, z) = exp(-(x - 50kilometers)^2 / (10kilometers)^2)

set!(model, η = ηᵢ)

uᵢ = XFaceField(grid)
U̅  = Field((Face, Center, Nothing), grid)

for i in 1:size(uᵢ, 1)
  uᵢ[i, :, :] .= - g * ∂xᶠᶜᶜ(i, 1, grid.Nz+1, grid, model.free_surface.η)
  U̅[i, 1, 1]   = - g * ∂xᶠᶜᶜ(i, 1, grid.Nz+1, grid, model.free_surface.η) * grid.Lz
end

set!(model, u = uᵢ)
set!(model.free_surface.state.U̅, U̅)

using Oceananigans.BoundaryConditions

fill_halo_regions!(model.velocities)

using Oceananigans.Utils
using Oceananigans.Models.HydrostaticFreeSurfaceModels: _update_zstar_split_explicit_scaling!
using Oceananigans.Models.HydrostaticFreeSurfaceModels: compute_w_from_continuity!, _update_z_star!

# # Scaling 
# s⁻    = model.grid.Δzᵃᵃᶠ.s⁻
# sⁿ    = model.grid.Δzᵃᵃᶠ.sⁿ
# ∂t_∂s = model.grid.Δzᵃᵃᶠ.∂t_∂s

# # Generalized vertical spacing
# ΔzF  = model.grid.Δzᵃᵃᶠ.Δ
# ΔzC  = model.grid.Δzᵃᵃᶜ.Δ

# # Reference (non moving) spacing
# ΔzF₀ = model.grid.Δzᵃᵃᶠ.Δr
# ΔzC₀ = model.grid.Δzᵃᵃᶜ.Δr

# launch!(architecture(model.grid), model.grid, :xy, _update_zstar_split_explicit_scaling!,
#   sⁿ, s⁻, ∂t_∂s, model.free_surface.η, U̅, ZeroField(grid), model.grid)

# fill_halo_regions!((sⁿ, s⁻, ∂t_∂s))

# launch!(architecture(grid), grid, :xy, _update_z_star!, 
#         ΔzF, ΔzC, ΔzF₀, ΔzC₀, sⁿ, Val(grid.Nz))
    
# fill_halo_regions!((ΔzF, ΔzC))

# compute_w_from_continuity!(model.velocities, CPU(), model.grid)

gravity_wave_speed   = sqrt(g * grid.Lz)
barotropic_time_step = grid.Δxᶜᵃᵃ / gravity_wave_speed

Δt = barotropic_time_step

@info "the time step is $Δt"

simulation = Simulation(model; Δt, stop_iteration = 1)

field_outputs = if model.grid isa ZStarSpacingGrid
  merge(model.velocities, model.tracers, (; ΔzF = model.grid.Δzᵃᵃᶠ.Δ))
else
  merge(model.velocities, model.tracers)
end

simulation.output_writers[:other_variables] = JLD2OutputWriter(model, field_outputs, 
                                                               overwrite_existing = true,
                                                               schedule = TimeInterval(30minutes),
                                                               filename = "zstar_model") 

function progress(sim)
    w  = interior(sim.model.velocities.w, :, :, sim.model.grid.Nz+1)
    u  = sim.model.velocities.u
    v  = sim.model.velocities.v
    η  = sim.model.free_surface.η
    
    msg0 = @sprintf("Time: %s iteration %d ", prettytime(sim.model.clock.time), sim.model.clock.iteration)
    msg1 = @sprintf("extrema w: %.2e %.2e ", maximum(w), minimum(w))
    msg2 = @sprintf("extrema u: %.2e %.2e ", maximum(u), minimum(u))
    if sim.model.grid isa ZStarSpacingGrid
      Δz = sim.model.grid.Δzᵃᵃᶠ.Δ
      msg4 = @sprintf("extrema Δz: %.2e %.2e ", maximum(Δz), minimum(Δz))
      @info msg0 * msg1 * msg2 * msg4
    else
      @info msg0 * msg1 * msg2
    end

    return nothing
end

simulation.callbacks[:progress] = Callback(progress, IterationInterval(1))
simulation.callbacks[:wizard]   = Callback(TimeStepWizard(; cfl = 0.2, max_change = 1.1), IterationInterval(10))
run!(simulation)
