
function initialize_free_surface!(sefs::SplitExplicitFreeSurface, grid, velocities)
    @apply_regionally compute_barotropic_mode!(sefs.state.U̅, sefs.state.V̅, grid, velocities.u, velocities.v)
    fill_halo_regions!((sefs.state.U̅, sefs.state.V̅, sefs.η))
end


# Barotropic Model Kernels
# u_Δz = u * Δz
@kernel function _barotropic_mode_kernel!(U, V, grid, ::Nothing, u, v)
    i, j  = @index(Global, NTuple)
    k_top = grid.Nz+1

    @inbounds U[i, j, k_top-1] = Δzᶠᶜᶜ(i, j, 1, grid) * u[i, j, 1]
    @inbounds V[i, j, k_top-1] = Δzᶜᶠᶜ(i, j, 1, grid) * v[i, j, 1]

    for k in 2:grid.Nz
        @inbounds U[i, j, k_top-1] += Δzᶠᶜᶜ(i, j, k, grid) * u[i, j, k]
        @inbounds V[i, j, k_top-1] += Δzᶜᶠᶜ(i, j, k, grid) * v[i, j, k]
    end
end

# Barotropic Model Kernels
# u_Δz = u * Δz
@kernel function _barotropic_mode_kernel!(U, V, grid, active_cells_map, u, v)
    idx = @index(Global, Linear)
    i, j = active_linear_index_to_tuple(idx, active_cells_map)
    k_top = grid.Nz+1

    @inbounds U[i, j, k_top-1] = Δzᶠᶜᶜ(i, j, 1, grid) * u[i, j, 1]
    @inbounds V[i, j, k_top-1] = Δzᶜᶠᶜ(i, j, 1, grid) * v[i, j, 1]

    for k in 2:grid.Nz
        @inbounds U[i, j, k_top-1] += Δzᶠᶜᶜ(i, j, k, grid) * u[i, j, k]
        @inbounds V[i, j, k_top-1] += Δzᶜᶠᶜ(i, j, k, grid) * v[i, j, k]
    end
end

@inline function compute_barotropic_mode!(U, V, grid, u, v)
    active_cells_map = retrieve_surface_active_cells_map(grid)

    launch!(architecture(grid), grid, :xy, _barotropic_mode_kernel!, U, V, grid, active_cells_map, u, v; active_cells_map)

    return nothing
end

@kernel function _barotropic_split_explicit_corrector!(u, v, U̅, V̅, U, V, grid)
    i, j, k = @index(Global, NTuple)
    k_top = grid.Nz+1

    @inbounds begin
        Hᶠᶜ = static_column_depthᶠᶜᵃ(i, j, grid)
        Hᶜᶠ = static_column_depthᶜᶠᵃ(i, j, grid)
        
        u[i, j, k] = u[i, j, k] + (U̅[i, j, k_top-1] - U[i, j, k_top-1]) / Hᶠᶜ
        v[i, j, k] = v[i, j, k] + (V̅[i, j, k_top-1] - V[i, j, k_top-1]) / Hᶜᶠ
    end
end

function barotropic_split_explicit_corrector!(u, v, free_surface, grid)
    sefs       = free_surface.state
    U, V, U̅, V̅ = sefs.U, sefs.V, sefs.U̅, sefs.V̅
    arch       = architecture(grid)

    # take out "bad" barotropic mode, 
    # !!!! reusing U and V for this storage since last timestep doesn't matter
    compute_barotropic_mode!(U, V, grid, u, v)
    # add in "good" barotropic mode
    launch!(arch, grid, :xyz, _barotropic_split_explicit_corrector!,
            u, v, U̅, V̅, U, V, grid)

    return nothing
end
