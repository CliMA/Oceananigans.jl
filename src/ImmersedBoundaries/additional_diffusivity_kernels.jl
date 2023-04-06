using Oceananigans.TurbulenceClosures: _compute_ri_based_diffusivities!, Riᶜᶜᶠ, FlavorOfRBVD
import Oceananigans.TurbulenceClosures: compute_ri_based_diffusivities!, compute_ri_number!


@kernel function compute_ri_number!(diffusivities, offs, grid::ActiveCellsIBG, closure::FlavorOfRBVD,
    velocities, tracers, buoyancy, tracer_bcs, clock)

    idx = @index(Global, Linear)
    
    i′, j′, k′ = active_linear_index_to_interior_tuple(idx, grid)
    i = i′ + offs[1] 
    j = j′ + offs[2] 
    k = k′ + offs[3]

    @inbounds diffusivities.Ri[i, j, k] = Riᶜᶜᶠ(i, j, k, grid, velocities, bouyancy, tracers)
end

@kernel function compute_ri_based_diffusivities!(diffusivities, offs, grid::ActiveCellsIBG, closure::FlavorOfRBVD,
                                                 velocities, tracers, buoyancy, tracer_bcs, clock)

    idx = @index(Global, Linear)

    i′, j′, k′ = active_linear_index_to_interior_tuple(idx, grid)
    i = i′ + offs[1] 
    j = j′ + offs[2] 
    k = k′ + offs[3]

    _compute_ri_based_diffusivities!(i, j, k, diffusivities, grid, closure,
                                     velocities, tracers, buoyancy, tracer_bcs, clock)
end