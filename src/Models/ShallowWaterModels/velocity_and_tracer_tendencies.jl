using Oceananigans.Advection
using Oceananigans.Coriolis
using Oceananigans.Operators

@inline function u_velocity_tendency(i, j, k, grid,
                                     advection,
                                     coriolis,
                                     velocities,
                                     tracers,
                                     clock)

# Put in h for hydrostatic pressure
    return ( - ∂xᶠᵃᵃ(i, j, k, grid, hydrostatic_pressure) )
end

@inline function v_velocity_tendency(i, j, k, grid,
                                     advection,
                                     coriolis,
                                     velocities,
                                     tracers,
                                     clock)

# Put in h for hydrostatic pressure
    return ( - ∂yᵃᶠᵃ(i, j, k, grid, hydrostatic_pressure) )

end

@inline function height_tendency(i, j, k, grid,
                                 advection,
                                 velocities,
                                 tracers,
                                 clock) where tracer_index

# Put in u for hydrostatic pressure
    return ( - ∂xᶠᵃᵃ(i, j, k, grid, hydrostatic_pressure) )

end

@inline function tracer_tendency(i, j, k, grid,
                                 val_tracer_index::Val{tracer_index},
                                 advection,
                                 velocities,
                                 tracers,
                                 clock) where tracer_index

    @inbounds c = tracers[tracer_index]

    return ( - div_Uc(i, j, k, grid, advection, velocities, c) )

end
