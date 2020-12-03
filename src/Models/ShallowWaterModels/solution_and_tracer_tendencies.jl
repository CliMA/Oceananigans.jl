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

    return ( - gravitational_acceleration * ∂xᶠᵃᵃ(i, j, k, grid, solution.h)
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

    return ( - gravitational_acceleration * ∂yᵃᶠᵃ(i, j, k, grid, solution.h)
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

    return ( - ∂xᶜᵃᵃ(i, j, k, grid, solution.uh) )
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
