# Kernels to compute the vertical integral of the velocities
@kernel function _compute_barotropic_mode!(U̅, V̅, grid, u, v, η)
    i, j  = @index(Global, NTuple)
    k_top  = size(grid, 3) + 1

    hᶠᶜ = static_column_depthᶠᶜᵃ(i, j, grid)
    hᶜᶠ = static_column_depthᶜᶠᵃ(i, j, grid)

    Hᶠᶜ = column_depthᶠᶜᵃ(i, j, k_top, grid, η)
    Hᶜᶠ = column_depthᶜᶠᵃ(i, j, k_top, grid, η)

    # If the static depths are zero (i.e. the column is immersed),
    # we set the grid scaling factor to 1
    # (There is no free surface on an immersed column (η == 0))
    σᶠᶜ = ifelse(hᶠᶜ == 0, one(grid), Hᶠᶜ / hᶠᶜ)
    σᶜᶠ = ifelse(hᶜᶠ == 0, one(grid), Hᶜᶠ / hᶜᶠ)

    @inbounds U̅[i, j, 1] = Δrᶠᶜᶜ(i, j, 1, grid) * u[i, j, 1] * σᶠᶜ
    @inbounds V̅[i, j, 1] = Δrᶜᶠᶜ(i, j, 1, grid) * v[i, j, 1] * σᶜᶠ

    for k in 2:grid.Nz
        @inbounds U̅[i, j, 1] += Δrᶠᶜᶜ(i, j, k, grid) * u[i, j, k] * σᶠᶜ
        @inbounds V̅[i, j, 1] += Δrᶜᶠᶜ(i, j, k, grid) * v[i, j, k] * σᶜᶠ
    end
end

# Note: this function is also used during initialization
function compute_barotropic_mode!(U̅, V̅, grid, u, v, η)
    active_cells_map = get_active_column_map(grid) # may be nothing

    launch!(architecture(grid), grid, :xy,
            _compute_barotropic_mode!,
            U̅, V̅, grid, u, v, η; active_cells_map)

    return nothing
end

# Correcting `u` and `v` with the barotropic mode computed in `free_surface`
function barotropic_split_explicit_corrector!(u, v, free_surface, grid)
    state = free_surface.filtered_state
    η     = free_surface.η
    U, V  = free_surface.barotropic_velocities
    U̅, V̅  = state.U, state.V
    arch  = architecture(grid)

    # NOTE: the filtered `U̅` and `V̅` have been copied in the instantaneous `U` and `V`,
    # so we use the filtered velocities as "work arrays" to store the vertical integrals
    # of the instantaneous velocities `u` and `v`.
    compute_barotropic_mode!(U̅, V̅, grid, u, v, η)

    # add in "good" barotropic mode
    launch!(arch, grid, :xyz, _barotropic_split_explicit_corrector!,
            u, v, U, V, U̅, V̅, η, grid)

    return nothing
end

@kernel function _barotropic_split_explicit_corrector!(u, v, U, V, U̅, V̅, η, grid)
    i, j, k = @index(Global, NTuple)
    k_top = size(grid, 3) + 1

    @inbounds begin
        Hᶠᶜ = column_depthᶠᶜᵃ(i, j, k_top, grid, η)
        Hᶜᶠ = column_depthᶜᶠᵃ(i, j, k_top, grid, η)

        u[i, j, k] = u[i, j, k] + (U[i, j, 1] - U̅[i, j, 1]) / Hᶠᶜ
        v[i, j, k] = v[i, j, k] + (V[i, j, 1] - V̅[i, j, 1]) / Hᶜᶠ
    end
end


