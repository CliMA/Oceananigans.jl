using Oceananigans.Advection
using Oceananigans.Coriolis
using Oceananigans.Operators

"""
Compute the tendency for the x-directional transport, uh
"""
@inline function uh_solution_tendency(i, j, k, grid,
                                      gravitational_acceleration,
                                      advection,
                                      coriolis,
                                      bathymetry,
                                      solution,
                                      tracers,
                                      diffusivities,
                                      forcings,
                                      clock)

    uh2 = solution.uh^2 
    termx = ℑxᶜᵃᵃ(i, j, k, grid, uh2) / solution.h + 0.5 * gravitational_acceleration * solution.h^2
    termy = ℑyᵃᶠᵃ(i, j, k, grid, solution.uh) * ℑxᶠᵃᵃ(i, j, k, grid, solution.vh) / ℑxyᶠᶠᵃ(i, j, k, grid, solution.h)
    
    return ( - ∂xᶠᵃᵃ(i, j, k, grid, termx)
             - ∂yᵃᶠᵃ(i, j, k, grid, termy)
             + coriolis.f * ℑxyᶠᶜᵃ(i, j, k, grid, solution.vh))

end

"""
Compute the tendency for the y-directional transport, vh.
"""
@inline function vh_solution_tendency(i, j, k, grid,
                                      gravitational_acceleration,
                                      advection,
                                      coriolis,
                                      bathymetry,
                                      solution,
                                      tracers,
                                      diffusivities,
                                      forcings,
                                      clock)

    vh2 = solution.vh^2
    termx = ℑyᵃᶠᵃ(i, j, k, grid, solution.uh) * ℑxᶠᵃᵃ(i, j, k, grid, solution.vh) / ℑxyᶠᶠᵃ(i, j, k, grid, solution.h)
    termy = ℑyᵃᶜᵃ(i, j, k, grid, vh2) / solution.h + 0.5 * gravitational_acceleration * solution.h^2
    
    return ( - ∂xᶠᵃᵃ(i, j, k, grid, termx)
             - ∂yᵃᶠᵃ(i, j, k, grid, termy)
             - coriolis.f * ℑxyᶜᶠᵃ(i, j, k, grid, solution.uh))

end

"""
Compute the tendency for the height, h.
"""
@inline function h_solution_tendency(i, j, k, grid,
                                     gravitational_acceleration,
                                     advection,
                                     coriolis,
                                     bathymetry,
                                     solution,
                                     tracers,
                                     diffusivities,
                                     forcings,
                                     clock) where tracer_index

    return ( - ∂xᶜᵃᵃ(i, j, k, grid, solution.uh)
             - ∂yᵃᶜᵃ(i, j, k, grid, solution.vh) )
end

@inline function tracer_tendency(i, j, k, grid,
                                 val_tracer_index::Val{tracer_index},
                                 advection,
                                 solution,
                                 tracers,
                                 diffusivities,
                                 forcings,
                                 clock) where tracer_index

    @inbounds c = tracers[tracer_index]

    return ( 0.0 )
end
