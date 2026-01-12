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

"""
    compute_barotropic_mode!(U̅, V̅, grid, u, v)

Compute the depth-integrated (barotropic) velocities from baroclinic velocity fields.

The barotropic transport is computed as: `U̅ = ∫ u dz` and `V̅ = ∫ v dz`.
This function is used both during split-explicit correction and initialization.
"""
function compute_barotropic_mode!(U̅, V̅, grid, u, v)
    active_cells_map = get_active_column_map(grid) # may be nothing

    launch!(architecture(grid), grid, :xy,
            _compute_barotropic_mode!,
            U̅, V̅, grid, u, v; active_cells_map)

    return nothing
end

"""
    barotropic_split_explicit_corrector!(u, v, free_surface, grid)

Correct baroclinic velocities so that they are consistent with the barotropic flow from
split-explicit substepping.

The correction ensures that the depth-integrated baroclinic velocity matches the
filtered barotropic velocity from the split-explicit scheme:

    u_corrected = u + (U_filtered - U_baroclinic) / H

where `U_filtered` is the filtered barotropic transport from substepping and
`U_baroclinic` is the depth-integral of the baroclinic velocity.
"""
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

@kernel function _compute_transport_velocities!(ũ, ṽ, grid, Ũ, Ṽ, u, v, U̅, V̅)
    i, j, k = @index(Global, NTuple)
    Hᶠᶜ = column_depthᶠᶜᵃ(i, j, grid)
    Hᶜᶠ = column_depthᶜᶠᵃ(i, j, grid)

    immersedᶜᶠᶜ = peripheral_node(i, j, k, grid, Center(), Face(), Center())
    immersedᶠᶜᶜ = peripheral_node(i, j, k, grid, Face(), Center(), Center())

    δuᵢ = @inbounds U[i, j, 1] - U̅[i, j, 1]
    δvⱼ = @inbounds V[i, j, 1] - V̅[i, j, 1]

    u_correction = ifelse(Hᶠᶜ == 0, zero(grid), δuᵢ / Hᶠᶜ)
    v_correction = ifelse(Hᶜᶠ == 0, zero(grid), δvⱼ / Hᶜᶠ)

    @inbounds begin
        ũ⁺ = u[i, j, k] + u_correction
        ṽ⁺ = v[i, j, k] + v_correction

        ũ[i, j, k] = ifelse(immersedᶠᶜᶜ, zero(grid), ũ⁺)
        ṽ[i, j, k] = ifelse(immersedᶜᶠᶜ, zero(grid), ṽ⁺)
    end
end

"""
    compute_transport_velocities!(model, free_surface::SplitExplicitFreeSurface)

Compute transport velocities used for tracer advection with split-explicit free surface.

Transport velocities differ from prognostic velocities by including the barotropic correction:

    ũ = u + (Ũ_filtered - U_baroclinic) / H

where `Ũ_filtered` is the time-filtered barotropic transport from split-explicit substepping.
This ensures that tracers are advected with a velocity field consistent with the filtered
free surface evolution.

After computing horizontal transport velocities, vertical transport velocity `w̃` is computed
from continuity and halo regions are filled.
"""
function compute_transport_velocities!(model, free_surface::SplitExplicitFreeSurface)
    grid = model.grid
    u, v, _ = model.velocities
    ũ, ṽ, _ = model.transport_velocities
    Ũ = free_surface.filtered_state.Ũ
    Ṽ = free_surface.filtered_state.Ṽ
    U̅ = free_surface.filtered_state.U̅
    V̅ = free_surface.filtered_state.V̅

    @apply_regionally begin
        compute_barotropic_mode!(U̅, V̅, grid, u, v)
        launch!(architecture(grid), grid, :xyz, _compute_transport_velocities!, ũ, ṽ, grid, Ũ, Ṽ, u, v, U̅, V̅)
    end

    # Fill transport velocities
    fill_halo_regions!((ũ, ṽ); async=true)

    # Update grid velocity and vertical transport velocity
    @apply_regionally update_vertical_velocities!(model.transport_velocities, model.grid, model)

    return nothing
end
