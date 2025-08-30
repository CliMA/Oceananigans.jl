using Oceananigans.ImmersedBoundaries: get_active_column_map, peripheral_node
using Oceananigans.TimeSteppers: QuasiAdamsBashforth2TimeStepper, SplitRungeKutta3TimeStepper
using Oceananigans.Operators: Δz

# This file contains two different initializations methods performed at different stages of the simulation.
#
# - `initialize_free_surface!`: the first initialization, performed only once at the beginning of the simulation,
#                               calculates the barotropic velocities from the velocity initial conditions.
#
# - `initialize_free_surface_state!`: is performed at the beginning of the substepping procedure, resets the filtered state to zero
#                                     and reinitializes the timestepper auxiliaries from the previous filtered state.

# `initialize_free_surface!` is called at the beginning of the simulation to initialize the free surface state
# from the initial velocity conditions.
function initialize_free_surface!(sefs::SplitExplicitFreeSurface, grid, velocities)
    barotropic_velocities = sefs.barotropic_velocities
    u, v, w = velocities
    @apply_regionally compute_barotropic_mode!(barotropic_velocities.U,
                                               barotropic_velocities.V,
                                               grid, u, v)

    fill_halo_regions!((barotropic_velocities.U, barotropic_velocities.V))
    fill_halo_regions!(sefs.η)

    return nothing
end

# `initialize_free_surface_state!` is called at the beginning of the substepping to
# reset the filtered state to zero and reinitialize the state from the filtered state.
function initialize_free_surface_state!(free_surface, baroclinic_timestepper, timestepper)

    η = free_surface.η
    U, V = free_surface.barotropic_velocities

    initialize_free_surface_timestepper!(timestepper, η, U, V)

    fill!(free_surface.filtered_state.η, 0)
    fill!(free_surface.filtered_state.U, 0)
    fill!(free_surface.filtered_state.V, 0)

    return nothing
end

# At the last stage we reset the velocities and perform the complete substepping from n to n+1
function initialize_free_surface_state!(free_surface, baroclinic_ts::SplitRungeKutta3TimeStepper, barotropic_ts)

    η = free_surface.η
    U, V = free_surface.barotropic_velocities

    Uⁿ⁻¹ = baroclinic_ts.Ψ⁻.U
    Vⁿ⁻¹ = baroclinic_ts.Ψ⁻.V
    ηⁿ⁻¹ = baroclinic_ts.Ψ⁻.η

    # Restart from the state at baroclinic step n
    parent(U) .= parent(Uⁿ⁻¹)
    parent(V) .= parent(Vⁿ⁻¹)
    parent(η) .= parent(ηⁿ⁻¹)

    initialize_free_surface_timestepper!(barotropic_ts, η, U, V)

    fill!(free_surface.filtered_state.η, 0)
    fill!(free_surface.filtered_state.U, 0)
    fill!(free_surface.filtered_state.V, 0)

    return nothing
end
