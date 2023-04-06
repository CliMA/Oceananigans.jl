using Oceananigans.TurbulenceClosures: _compute_ri_based_diffusivities!
import Oceananigans.TurbulenceClosures: compute_ri_based_diffusivities!

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