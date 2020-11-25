using Oceananigans.Advection
using Oceananigans.Coriolis
using Oceananigans.Operators

"""
Compute the tendency for the x-directional transport, uh
"""
@inline function uh_solution_tendency(i, j, k, grid,
                                      advection,
                                      coriolis,
                                      solution,
                                      tracers,
                                      clock)

    return ( - 0.0 )
#    return ( - ∂xᶠᵃᵃ(i, j, k, grid, solution.h) )
end

"""
Compute the tendency for the y-directional transport, vh.
"""
@inline function vh_solution_tendency(i, j, k, grid,
                                      advection,
                                      coriolis,
                                      solution,
                                      tracers,
                                      clock)

    return ( -0.0 )
#    return ( - ∂yᵃᶠᵃ(i, j, k, grid, solution.h) )

end

"""
Compute the tendency for the height, h.
"""
@inline function h_solution_tendency(i, j, k, grid,
                            advection,
                            solution,
                            tracers,
                            clock) where tracer_index

    return ( - 0.0  )
#    return ( - ∂xᶠᵃᵃ(i, j, k, grid, solution.h) )

end

@inline function tracer_tendency(i, j, k, grid,
                                 val_tracer_index::Val{tracer_index},
                                 advection,
                                 solution,
                                 tracers,
                                 clock) where tracer_index

    @inbounds c = tracers[tracer_index]

    # velocities should be replaced with transport/h, as a vector
    return ( 0.0 )
#    return ( - div_Uc(i, j, k, grid, advection, solution, c) )

end
