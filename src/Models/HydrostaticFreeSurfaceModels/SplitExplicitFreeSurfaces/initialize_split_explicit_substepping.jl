using Oceananigans.ImmersedBoundaries: retrieve_surface_active_cells_map, peripheral_node

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
    @apply_regionally compute_barotropic_mode!(barotropic_velocities.U, barotropic_velocities.V, grid, velocities.u, velocities.v)
    fill_halo_regions!((barotropic_velocities.U, barotropic_velocities.V))
    return nothing
end

# `initialize_free_surface_state!` is called at the beginning of the substepping to 
# reset the filtered state to zero and reinitialize the state from the filtered state.
function initialize_free_surface_state!(filtered_state, η, velocities, timestepper)

    initialize_free_surface_timestepper!(timestepper, η, velocities)

    fill!(filtered_state.η, 0)
    fill!(filtered_state.U, 0)
    fill!(filtered_state.V, 0)

    return nothing
end
