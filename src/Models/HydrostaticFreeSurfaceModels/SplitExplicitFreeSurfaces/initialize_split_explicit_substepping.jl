using Oceananigans.Grids: get_active_column_map, peripheral_node
using Oceananigans.TimeSteppers: QuasiAdamsBashforth2TimeStepper, SplitRungeKuttaTimeStepper

# This file contains two different methods performed at different stages of the simulation.
#
# - `reconcile_free_surface!`: reconciles the barotropic velocities with the 3D velocity fields.
#                              Called during `set!` and `initialize!` to ensure consistency.
#
# - `initialize_free_surface_state!`: is performed at the beginning of the substepping procedure, resets the filtered state to zero
#                                     and reinitializes the timestepper auxiliaries from the previous filtered state.

# `reconcile_free_surface!` computes the barotropic mode from velocity fields to ensure consistency.
function reconcile_free_surface!(sefs::SplitExplicitFreeSurface, grid, velocities)
    barotropic_velocities = sefs.barotropic_velocities
    u, v, w = velocities
    @apply_regionally compute_barotropic_mode!(barotropic_velocities.U,
                                               barotropic_velocities.V,
                                               grid, u, v)

    η = sefs.displacement
    U, V = barotropic_velocities.U, barotropic_velocities.V
    barotropic_model_fields = (; U, V, η)
    # fill_halo_regions!((U, V), nothing, barotropic_model_fields)
    fill_halo_regions!(η, nothing, barotropic_model_fields)

    return nothing
end

# `initialize_free_surface_state!` is called at the beginning of the substepping to reset the filtered state to zero and
# reinitialize the state from the filtered state.
function initialize_free_surface_state!(free_surface, baroclinic_timestepper, timestepper)

    η = free_surface.displacement
    U, V = free_surface.barotropic_velocities

    initialize_free_surface_timestepper!(timestepper, η, U, V)

    for field in free_surface.filtered_state
        fill!(field, 0)
    end

    return nothing
end

# At the last stage we reset the velocities and perform the complete substepping from n to n+1
function initialize_free_surface_state!(free_surface, baroclinic_ts::SplitRungeKuttaTimeStepper, barotropic_ts)

    η = free_surface.displacement
    U, V = free_surface.barotropic_velocities

    Uⁿ⁻¹ = baroclinic_ts.Ψ⁻.U
    Vⁿ⁻¹ = baroclinic_ts.Ψ⁻.V
    ηⁿ⁻¹ = baroclinic_ts.Ψ⁻.η

    # Restart from the state at baroclinic step n
    parent(U) .= parent(Uⁿ⁻¹)
    parent(V) .= parent(Vⁿ⁻¹)
    parent(η) .= parent(ηⁿ⁻¹)

    initialize_free_surface_timestepper!(barotropic_ts, η, U, V)

    for field in free_surface.filtered_state
        fill!(field, 0)
    end

    return nothing
end
