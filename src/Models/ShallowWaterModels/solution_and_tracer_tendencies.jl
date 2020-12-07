using Oceananigans.Advection
using Oceananigans.Coriolis
using Oceananigans.Operators

@inline squared(i, j, k, grid, ϕ) = ϕ[i, j, k]^2

@inline uh_transport_x(i, j, k, grid, h, uh, g) =
    @inbounds ℑxᶜᵃᵃ(i, j, k, grid, squared, uh) / h[i, j, k] + g/2 * h[i, j, k]^2

@inline uh_transport_y(i, j, k, grid, h, uh, vh) =
    ℑyᵃᶠᵃ(i, j, k, grid, uh) * ℑxᶠᵃᵃ(i, j, k, grid, vh) / ℑxyᶠᶠᵃ(i, j, k, grid, h)

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

#    termx = ℑxᶜᵃᵃ(i, j, k, grid, squared, solution.uh) / solution.h + 0.5 * gravitational_acceleration * solution.h^2
#    termy = ℑyᵃᶠᵃ(i, j, k, grid, solution.uh) * ℑxᶠᵃᵃ(i, j, k, grid, solution.vh) / ℑxyᶠᶠᵃ(i, j, k, grid, solution.h)

#    return ( - ∂xᶠᵃᵃ(i, j, k, grid, termx)
#             - ∂yᵃᶠᵃ(i, j, k, grid, termy)
    return ( - ∂xᶠᵃᵃ(i, j, k, grid, uh_transport_x, solution.h, solution.uh, gravitational_acceleration)
             - ∂yᵃᶠᵃ(i, j, k, grid, uh_transport_y, solution.h, solution.uh, solution.vh)
             + coriolis.f * ℑxyᶠᶜᵃ(i, j, k, grid, solution.vh))
end

@inline vh_transport_x(i, j, k, grid, h, uh, vh) =
    ℑyᵃᶠᵃ(i, j, k, grid, uh) * ℑxᶠᵃᵃ(i, j, k, grid, vh) / ℑxyᶠᶠᵃ(i, j, k, grid, h)

@inline vh_transport_y(i, j, k, grid, h, vh, g) =
    @inbounds ℑyᵃᶜᵃ(i, j, k, grid, squared, vh) / h[i, j, k] + g/2 * h[i, j, k]^2

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

#    termx = ℑyᵃᶠᵃ(i, j, k, grid, solution.uh) * ℑxᶠᵃᵃ(i, j, k, grid, solution.vh) / ℑxyᶠᶠᵃ(i, j, k, grid, solution.h)
#    termy = ℑyᵃᶜᵃ(i, j, k, grid, squared, solution.vh) / solution.h + 0.5 * gravitational_acceleration * solution.h^2

#    return ( - ∂xᶠᵃᵃ(i, j, k, grid, termx)
#             - ∂yᵃᶠᵃ(i, j, k, grid, termy)
    return ( - ∂xᶠᵃᵃ(i, j, k, grid, vh_transport_x, solution.h, solution.uh, solution.vh)
             - ∂yᵃᶠᵃ(i, j, k, grid, vh_transport_y, solution.h, solution.vh, gravitational_acceleration)
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
                                     clock)

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
