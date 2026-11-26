using Oceananigans.Grids: surface_kernel_parameters, volume_kernel_parameters
using Oceananigans.ImmersedBoundaries: immersed_inactive_node, immersed_peripheral_node

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
$(TYPEDSIGNATURES)

Compute the depth-integrated (barotropic) velocities from baroclinic velocity fields.

The barotropic transport is computed as: `U̅ = ∫ u dz` and `V̅ = ∫ v dz`.
This function is used both during split-explicit correction and initialization.
"""
function compute_barotropic_mode!(U̅, V̅, grid, u, v)
    launch!(architecture(grid), grid, surface_kernel_parameters(grid),
            _compute_barotropic_mode!,
            U̅, V̅, grid, u, v)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Project the baroclinic velocities onto the barotropic transport produced by the substepping,

    u = u + (U - ∫u dz) / H
"""
function barotropic_split_explicit_corrector!(u, v, free_surface, grid)
    state = free_surface.filtered_state
    U, V  = free_surface.barotropic_velocities
    U̅, V̅  = state.U̅, state.V̅
    arch  = architecture(grid)

    mask_immersed_field!(u)
    mask_immersed_field!(v)

    compute_barotropic_mode!(U̅, V̅, grid, u, v)
    launch!(arch, grid, volume_kernel_parameters(grid), _barotropic_split_explicit_corrector!, u, v, U, V, U̅, V̅, grid)

    return nothing
end

@kernel function _barotropic_split_explicit_corrector!(u, v, U, V, U̅, V̅, grid)
    i, j, k = @index(Global, NTuple)
    Hᶠᶜ = column_depthᶠᶜᵃ(i, j, grid)
    Hᶜᶠ = column_depthᶜᶠᵃ(i, j, grid)

    immersedᶠᶜᶜ = immersed_peripheral_node(i, j, k, grid, Face(), Center(), Center()) | immersed_inactive_node(i, j, k, grid, Face(), Center(), Center())
    immersedᶜᶠᶜ = immersed_peripheral_node(i, j, k, grid, Center(), Face(), Center()) | immersed_inactive_node(i, j, k, grid, Center(), Face(), Center())

    δuᵢ = @inbounds U[i, j, 1] - U̅[i, j, 1]
    δvⱼ = @inbounds V[i, j, 1] - V̅[i, j, 1]

    u_correction = ifelse(Hᶠᶜ == 0, zero(grid), δuᵢ / Hᶠᶜ)
    v_correction = ifelse(Hᶜᶠ == 0, zero(grid), δvⱼ / Hᶜᶠ)

    @inbounds u[i, j, k] = ifelse(immersedᶠᶜᶜ, zero(grid), u[i, j, k] + u_correction)
    @inbounds v[i, j, k] = ifelse(immersedᶜᶠᶜ, zero(grid), v[i, j, k] + v_correction)
end

@kernel function _compute_split_explicit_transport_velocities!(ũ, ṽ, grid, Ũ, Ṽ, u, v, U̅, V̅)
    i, j, k = @index(Global, NTuple)
    Hᶠᶜ = column_depthᶠᶜᵃ(i, j, grid)
    Hᶜᶠ = column_depthᶜᶠᵃ(i, j, grid)

    immersedᶜᶠᶜ = immersed_peripheral_node(i, j, k, grid, Center(), Face(), Center()) | immersed_inactive_node(i, j, k, grid, Center(), Face(), Center())
    immersedᶠᶜᶜ = immersed_peripheral_node(i, j, k, grid, Face(), Center(), Center()) | immersed_inactive_node(i, j, k, grid, Face(), Center(), Center())

    δuᵢ = @inbounds Ũ[i, j, 1] - U̅[i, j, 1]
    δvⱼ = @inbounds Ṽ[i, j, 1] - V̅[i, j, 1]

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
$(TYPEDSIGNATURES)

Compute transport velocities used for tracer advection with split-explicit free surface.

Transport velocities differ from prognostic velocities by including the barotropic correction:

    u = u + (Ũ - ∫udz) / H

where `Ũ` is the time-filtered barotropic transport from split-explicit substepping.
This ensures that tracers are advected with a velocity field consistent with the filtered
free surface evolution.

After computing horizontal transport velocities, vertical transport velocity `w̃` is computed
from continuity and halo regions are filled.
"""
function HydrostaticFreeSurfaceModels.compute_transport_velocities!(model, free_surface::SplitExplicitFreeSurface)
    grid = model.grid
    # `transport_velocities` holds the pre-step velocities and is also the destination; the kernel adds a
    # depth-uniform correction, so source and destination may alias.
    u = model.transport_velocities.u
    v = model.transport_velocities.v
    ũ, ṽ, w̃ = model.transport_velocities
    Ũ = free_surface.filtered_state.Ũ
    Ṽ = free_surface.filtered_state.Ṽ
    U̅ = free_surface.filtered_state.U̅
    V̅ = free_surface.filtered_state.V̅

    synchronize_communication!(Ũ)
    synchronize_communication!(Ṽ)

    compute_barotropic_mode!(U̅, V̅, grid, u, v)
    launch!(architecture(grid), grid, volume_kernel_parameters(grid),
            _compute_split_explicit_transport_velocities!,
            ũ, ṽ, grid, Ũ, Ṽ, u, v, U̅, V̅)

    update_vertical_velocities!(model.transport_velocities, model.grid, model)

    return nothing
end

#####
##### Correction of the radiating open boundary columns
#####

"""
$(TYPEDSIGNATURES)

Project the boundary-normal velocity columns of radiating open boundaries back onto the barotropic transport.
Radiation schemes step each level on its own, so after the boundary step these columns no longer integrate to
`U` and `V`. Other boundaries and the interior already satisfy the correction and are left untouched.
"""
function radiating_boundary_split_explicit_corrector!(u, v, free_surface)
    U, V = free_surface.barotropic_velocities
    correct_boundary_columns!(u.boundary_conditions.west,  West(),  u, U)
    correct_boundary_columns!(u.boundary_conditions.east,  East(),  u, U)
    correct_boundary_columns!(v.boundary_conditions.south, South(), v, V)
    correct_boundary_columns!(v.boundary_conditions.north, North(), v, V)
    return nothing
end

correct_boundary_columns!(bc, side, velocity, transport) = nothing

correct_boundary_columns!(::RNFBC, ::West,  u, U) = launch!(architecture(u.grid), u.grid, (size(u.grid, 2),), _correct_x_boundary_columns!, u, U, u.grid, 1)
correct_boundary_columns!(::RNFBC, ::East,  u, U) = launch!(architecture(u.grid), u.grid, (size(u.grid, 2),), _correct_x_boundary_columns!, u, U, u.grid, size(u.grid, 1) + 1)
correct_boundary_columns!(::RNFBC, ::South, v, V) = launch!(architecture(v.grid), v.grid, (size(v.grid, 1),), _correct_y_boundary_columns!, v, V, v.grid, 1)
correct_boundary_columns!(::RNFBC, ::North, v, V) = launch!(architecture(v.grid), v.grid, (size(v.grid, 1),), _correct_y_boundary_columns!, v, V, v.grid, size(v.grid, 2) + 1)

@inline peripheral_or_inactiveᶠᶜᶜ(i, j, k, grid) = immersed_peripheral_node(i, j, k, grid, Face(), Center(), Center()) | immersed_inactive_node(i, j, k, grid, Face(), Center(), Center())
@inline peripheral_or_inactiveᶜᶠᶜ(i, j, k, grid) = immersed_peripheral_node(i, j, k, grid, Center(), Face(), Center()) | immersed_inactive_node(i, j, k, grid, Center(), Face(), Center())

@kernel function _correct_x_boundary_columns!(u, U, grid, i)
    j = @index(Global, Linear)
    Hᶠᶜ = column_depthᶠᶜᵃ(i, j, grid)

    ∫u = zero(grid)
    for k in 1:grid.Nz
        @inbounds ∫u += ifelse(peripheral_or_inactiveᶠᶜᶜ(i, j, k, grid), zero(grid), Δzᶠᶜᶜ(i, j, k, grid) * u[i, j, k])
    end

    δu = @inbounds ifelse(Hᶠᶜ == 0, zero(grid), (U[i, j, 1] - ∫u) / Hᶠᶜ)

    for k in 1:grid.Nz
        @inbounds u[i, j, k] = ifelse(peripheral_or_inactiveᶠᶜᶜ(i, j, k, grid), zero(grid), u[i, j, k] + δu)
    end
end

@kernel function _correct_y_boundary_columns!(v, V, grid, j)
    i = @index(Global, Linear)
    Hᶜᶠ = column_depthᶜᶠᵃ(i, j, grid)

    ∫v = zero(grid)
    for k in 1:grid.Nz
        @inbounds ∫v += ifelse(peripheral_or_inactiveᶜᶠᶜ(i, j, k, grid), zero(grid), Δzᶜᶠᶜ(i, j, k, grid) * v[i, j, k])
    end

    δv = @inbounds ifelse(Hᶜᶠ == 0, zero(grid), (V[i, j, 1] - ∫v) / Hᶜᶠ)

    for k in 1:grid.Nz
        @inbounds v[i, j, k] = ifelse(peripheral_or_inactiveᶜᶠᶜ(i, j, k, grid), zero(grid), v[i, j, k] + δv)
    end
end
