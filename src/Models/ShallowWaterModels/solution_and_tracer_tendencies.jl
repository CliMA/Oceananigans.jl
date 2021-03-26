using Oceananigans.Advection
using Oceananigans.Coriolis
using Oceananigans.Operators

@inline squared(i, j, k, grid, ϕ) = @inbounds ϕ[i, j, k]^2

@inline gh2(i, j, k, grid, h, g) = @inbounds g/2 * h[i, j, k]^2

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

    g = gravitational_acceleration

    return ( - div_hUu(i, j, k, grid, advection, solution)
             - ∂xᶠᵃᵃ(i, j, k, grid, gh2, solution.h, gravitational_acceleration)
             - x_f_cross_U(i, j, k, grid, coriolis, solution)
             + forcings.uh(i, j, k, grid, clock, merge(solution, tracers)))
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

     g = gravitational_acceleration

    return ( - div_hUv(i, j, k, grid, advection, solution)
             - ∂yᵃᶠᵃ(i, j, k, grid, gh2, solution.h, gravitational_acceleration)
             - y_f_cross_U(i, j, k, grid, coriolis, solution)
             + forcings.vh(i, j, k, grid, clock, merge(solution, tracers)))
end

"""
Compute the tendency for the height, h.
"""
@inline function h_solution_tendency(i, j, k, grid,
                                     gravitational_acceleration,
                                     coriolis,
                                     bathymetry,
                                     solution,
                                     tracers,
                                     diffusivities,
                                     forcings,
                                     clock)

    return ( - div_uhvh(i, j, k, grid, solution)
             + forcings.h(i, j, k, grid, clock, merge(solution, tracers)))
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

    u = solution.uh / ℑxyᶠᶠᵃ(i, j, k, grid, solution.h)   # are these computed correctly?
    v = solution.vh / ℑxyᶠᶠᵃ(i, j, k, grid, solution.h)

    return ( 0.0
             - div_ucvc(i, j, k, grid, u, v, c)           # are these evaluated at the correct locations?
             + c * ∂xᶜᵃᵃ(i, j, k, grid, u)
             + c * ∂yᵃᶜᵃ(i, j, k, grid, v) 
            )
end
