# Kernels to compute the vertical integral of the velocities
@kernel function _barotropic_mode_kernel!(U, V, grid, ::Nothing, u, v)
    i, j  = @index(Global, NTuple)
    barotropic_mode_kernel!(U, V, i, j, grid, u, v)
end

@kernel function _barotropic_mode_kernel!(U, V, grid, active_cells_map, u, v)
    idx = @index(Global, Linear)
    i, j = active_linear_index_to_tuple(idx, active_cells_map)
    barotropic_mode_kernel!(U, V, i, j, grid, u, v)
end

@inline function barotropic_mode_kernel!(U, V, i, j, grid, u, v)
    @inbounds U[i, j, 1] = Δzᶠᶜᶜ(i, j, 1, grid) * u[i, j, 1]
    @inbounds V[i, j, 1] = Δzᶜᶠᶜ(i, j, 1, grid) * v[i, j, 1]

    for k in 2:grid.Nz
        @inbounds U[i, j, 1] += Δzᶠᶜᶜ(i, j, k, grid) * u[i, j, k]
        @inbounds V[i, j, 1] += Δzᶜᶠᶜ(i, j, k, grid) * v[i, j, k]
    end

    return nothing
end

@inline function compute_barotropic_mode!(U, V, grid, u, v)
    active_cells_map = retrieve_surface_active_cells_map(grid)
    launch!(architecture(grid), grid, :xy, _barotropic_mode_kernel!, U, V, grid, active_cells_map, u, v; active_cells_map)
    return nothing
end

@kernel function _barotropic_split_explicit_corrector!(u, v, U, V, U̅, V̅, grid)
    i, j, k = @index(Global, NTuple)

    @inbounds begin
        Hᶠᶜ = static_column_depthᶠᶜᵃ(i, j, grid)
        Hᶜᶠ = static_column_depthᶜᶠᵃ(i, j, grid)
        
        u[i, j, k] = u[i, j, k] + (U[i, j, 1] - U̅[i, j, 1]) / Hᶠᶜ
        v[i, j, k] = v[i, j, k] + (V[i, j, 1] - V̅[i, j, 1]) / Hᶜᶠ
    end
end

# Correcting `u` and `v` with the barotropic mode computed in `free_surface`
function barotropic_split_explicit_corrector!(u, v, free_surface, grid)
    state = free_surface.filtered_state
    U, V  = free_surface.barotropic_velocities
    U̅, V̅  = state.U, state.V
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
