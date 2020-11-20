using Oceananigans.Advection
using Oceananigans.Coriolis
using Oceananigans.Operators

@inline function uh_velocity_tendency(i, j, k, grid,
                                      advection,
                                      coriolis,
                                      solution,
                                      tracers,
                                      clock)

    return ( - ∂xᶠᵃᵃ(i, j, k, grid, solution.h) )
end

@inline function vh_velocity_tendency(i, j, k, grid,
                                      advection,
                                      coriolis,
                                      solution,
                                      tracers,
                                      clock)

    return ( - ∂yᵃᶠᵃ(i, j, k, grid, solution.h) )

end

@inline function h_tendency(i, j, k, grid,
                            advection,
                            solution,
                            tracers,
                            clock) where tracer_index

    return ( - ∂xᶠᵃᵃ(i, j, k, grid, solution.h) )

end

@inline function tracer_tendency(i, j, k, grid,
                                 val_tracer_index::Val{tracer_index},
                                 advection,
                                 solution,
                                 tracers,
                                 clock) where tracer_index

    @inbounds c = tracers[tracer_index]

    # velocities should be replaced with transport/h, as a vector
    return ( - div_Uc(i, j, k, grid, advection, solution, c) )

end
