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
    launch!(architecture(grid), grid, :xy,
            _compute_barotropic_mode!,
            U̅, V̅, grid, u, v; region=:interior)

    return nothing
end

# Correcting `u` and `v` with the barotropic mode computed in `free_surface`
function barotropic_split_explicit_corrector!(u, v, free_surface, grid)
    state = free_surface.filtered_state
    η     = free_surface.displacement
    U, V  = free_surface.barotropic_velocities
    U̅, V̅  = state.U̅, state.V̅
    arch  = architecture(grid)

    # Preparing velocities for the barotropic correction
    mask_immersed_field!(u)
    mask_immersed_field!(v)

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
    Hᶠᶜ = column_depthᶠᶜᵃ(i, j, grid)
    Hᶜᶠ = column_depthᶜᶠᵃ(i, j, grid)

    δuᵢ = @inbounds U[i, j, 1] - U̅[i, j, 1]
    δvⱼ = @inbounds V[i, j, 1] - V̅[i, j, 1]

    u_correction = ifelse(Hᶠᶜ == 0, zero(grid), δuᵢ / Hᶠᶜ)
    v_correction = ifelse(Hᶜᶠ == 0, zero(grid), δvⱼ / Hᶜᶠ)

    @inbounds u[i, j, k] = u[i, j, k] + u_correction
    @inbounds v[i, j, k] = v[i, j, k] + v_correction
end
