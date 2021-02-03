using Oceananigans.Coriolis
using Oceananigans.Operators: div_xyᶜᶜᵃ, Azᵃᵃᵃ, Vᵃᵃᶜ, δxᶠᵃᵃ, δyᵃᶜᵃ, δzᵃᵃᶜ



"""
Compute the tendency for the x-directional transport, uh
"""
@inline function shallow_water_x_momentum_tendency(i, j, k, grid,
                                                   solution::PrimitiveSolutionLinearizedHeightFields,
                                                   gravitational_acceleration,
                                                   advection,
                                                   coriolis,
                                                   bathymetry,
                                                   tracers,
                                                   diffusivities,
                                                   forcings,
                                                   clock)

    g = gravitational_acceleration

    return ( - U_dot_grad_u(i, j, k, grid, solution.u, solution.v)
             - g * ∂xᶠᵃᵃ(i, j, k, grid, solution.η)
             - x_f_cross_U(i, j, k, grid, coriolis, solution)
             + forcings.u(i, j, k, grid, clock, merge(solution, tracers))) end

"""
Compute the tendency for the y-directional transport, vh.
"""
@inline function shallow_water_y_momentum_tendency(i, j, k, grid,
                                                   solution::PrimitiveSolutionLinearizedHeightFields,
                                                   gravitational_acceleration,
                                                   advection,
                                                   coriolis,
                                                   bathymetry,
                                                   tracers,
                                                   diffusivities,
                                                   forcings,
                                                   clock)

     g = gravitational_acceleration

    return ( - U_dot_grad_v(i, j, k, grid, solution.u, solution.v)
             - g * ∂xᶠᵃᵃ(i, j, k, grid, solution.η)
             - y_f_cross_U(i, j, k, grid, coriolis, solution)
             + forcings.v(i, j, k, grid, clock, merge(solution, tracers)))
end

"""
Compute the tendency for the height, h.
"""
@inline function shallow_water_height_tendency(i, j, k, grid,
                                               solution::PrimitiveSolutionLinearizedHeightFields,
                                               gravitational_acceleration,
                                               advection,
                                               coriolis,
                                               bathymetry,
                                               tracers,
                                               diffusivities,
                                               forcings,
                                               clock)

    return ( - grid.Lz * div_xyᶜᶜᵃ(i, j, k, grid, solution.u, solution.v)
             + forcings.η(i, j, k, grid, clock, merge(solution, tracers)))
end

@inline function shallow_water_tracer_tendency(i, j, k, grid,
                                               solution::PrimitiveSolutionLinearizedHeightFields,
                                               val_tracer_index::Val{tracer_index},
                                               advection,
                                               tracers,
                                               diffusivities,
                                               forcings,
                                               clock) where tracer_index

    @inbounds c = tracers[tracer_index]

    return - div_Uc(i, j, k, grid, advection, solution, c)
end
