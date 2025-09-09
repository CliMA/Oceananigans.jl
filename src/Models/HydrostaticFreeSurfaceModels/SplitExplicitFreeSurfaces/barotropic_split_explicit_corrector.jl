# Kernels to compute the vertical integral of the velocities
@kernel function _compute_barotropic_mode!(U̅, V̅, grid, u, v)
    i, j  = @index(Global, NTuple)

    @inbounds U̅[i, j, 1] = Δzᶠᶜᶜ(i, j, 1, grid) * u[i, j, 1] 
    @inbounds V̅[i, j, 1] = Δzᶜᶠᶜ(i, j, 1, grid) * v[i, j, 1] 

    for k in 2:grid.Nz
        @inbounds U̅[i, j, 1] += Δzᶠᶜᶜ(i, j, k, grid) * u[i, j, k] 
        @inbounds V̅[i, j, 1] += Δzᶜᶠᶜ(i, j, k, grid) * v[i, j, k] 
    end
end

# Note: this function is also used during initialization
function compute_barotropic_mode!(U̅, V̅, grid, u, v)
    active_cells_map = get_active_column_map(grid) # may be nothing

    launch!(architecture(grid), grid, :xy,
            _compute_barotropic_mode!,
            U̅, V̅, grid, u, v; active_cells_map)

    return nothing
end

# Correcting `u` and `v` with the barotropic mode computed in `free_surface`
function barotropic_split_explicit_corrector!(u, v, free_surface, grid)
    state = free_surface.filtered_state
    η     = free_surface.η
    U, V  = free_surface.barotropic_velocities
    U̅, V̅  = state.U̅, state.V̅
    arch  = architecture(grid)

    # NOTE: the filtered `U̅` and `V̅` have been copied in the instantaneous `U` and `V`,
    # so we use the filtered velocities as "work arrays" to store the vertical integrals
    # of the instantaneous velocities `u` and `v`.
    compute_barotropic_mode!(U̅, V̅, grid, u, v)

    # add in "good" barotropic mode
    launch!(arch, grid, :xyz, _barotropic_split_explicit_corrector!,
            u, v, U, V, U̅, V̅, grid)

    return nothing
end

@kernel function _barotropic_split_explicit_corrector!(u, v, U, V, U̅, V̅, grid)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        Hᶠᶜ = column_depthᶠᶜᵃ(i, j, grid)
        Hᶜᶠ = column_depthᶜᶠᵃ(i, j, grid)

        u[i, j, k] = u[i, j, k] + (U[i, j, 1] - U̅[i, j, 1]) / Hᶠᶜ
        v[i, j, k] = v[i, j, k] + (V[i, j, 1] - V̅[i, j, 1]) / Hᶜᶠ
    end
end

@kernel function _compute_transport_velocities!(ũ, ṽ, grid, Ũ, Ṽ, u, v, U̅, V̅)
    i, j = @index(Global, NTuple)
    
    for k in 1:size(grid, 3)
        @inline ũ[i, j, k] = u[i, j, k] + (Ũ[i, j, 1] - U̅[i, j, 1]) / column_depthᶠᶜᵃ(i, j, grid)
        @inline ṽ[i, j, k] = v[i, j, k] + (Ṽ[i, j, 1] - V̅[i, j, 1]) / column_depthᶜᶠᵃ(i, j, grid)
    end
end

function compute_transport_velocities!(model, free_surface::SplitExplicitFreeSurface)
    grid = model.grid
    u, v, _ = model.velocities
    ũ, ṽ, _ = model.transport_velocities
    Ũ = free_surface.filtered_state.Ũ
    Ṽ = free_surface.filtered_state.Ṽ
    U̅ = free_surface.filtered_state.U̅
    V̅ = free_surface.filtered_state.V̅

    compute_barotropic_mode!(U̅, V̅, grid, u, v)

    launch!(architecture(grid), grid, :xy,
            _compute_transport_velocities!, ũ, ṽ, grid, Ũ, Ṽ, u, v, U̅, V̅)

    # Fill barotropic stuff...
    fill_halo_regions!((ũ, ṽ); async=true)
    
    # Update grid velocity and vertical transport velocity
    update_vertical_velocities!(model.transport_velocities, model.grid, model)

    return nothing
end