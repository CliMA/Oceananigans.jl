using Oceananigans.ImmersedBoundaries: immersed_peripheral_node, immersed_inactive_node
using Oceananigans.Grids: peripheral_node
using Oceananigans.Models: surface_kernel_parameters, volume_kernel_parameters

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

# The mean, not the transport: a mutable coordinate rescales the column before the correction reads this.
# Runs before the velocities are masked, hence the explicit `peripheral_node` guard.
@kernel function _compute_barotropic_velocity!(ū, v̄, grid, u, v)
    i, j = @index(Global, NTuple)
    k_top = grid.Nz + 1

    Hᶠᶜ = column_depthᶠᶜᵃ(i, j, k_top, grid)
    Hᶜᶠ = column_depthᶜᶠᵃ(i, j, k_top, grid)

    U = zero(grid)
    V = zero(grid)

    for k in 1:grid.Nz
        @inbounds U += Δzᶠᶜᶜ(i, j, k, grid) * u[i, j, k] * !peripheral_node(i, j, k, grid, Face(), Center(), Center())
        @inbounds V += Δzᶜᶠᶜ(i, j, k, grid) * v[i, j, k] * !peripheral_node(i, j, k, grid, Center(), Face(), Center())
    end

    @inbounds ū[i, j, 1] = ifelse(Hᶠᶜ > 0, U / Hᶠᶜ, zero(grid))
    @inbounds v̄[i, j, 1] = ifelse(Hᶜᶠ > 0, V / Hᶜᶠ, zero(grid))
end

"""
$(TYPEDSIGNATURES)

Store the depth-mean velocities the vertical solver is about to act on, against which the correction is
measured. Without an implicit boundary flux the solver conserves the column integral and there is nothing
to preserve.
"""
function store_pre_solve_velocities!(model, free_surface::SplitExplicitFreeSurface)
    cᵁ, cⱽ = barotropic_boundary_coefficients(free_surface.implicit_boundary_coefficients)
    isnothing(cᵁ) && isnothing(cⱽ) && return nothing

    state = free_surface.filtered_state
    u, v, _ = model.velocities
    grid = model.grid
    launch!(architecture(grid), grid, surface_kernel_parameters(grid),
            _compute_barotropic_velocity!, state.U̅, state.V̅, grid, u, v)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Correct baroclinic velocities so that they are consistent with the barotropic flow from split-explicit substepping,

    u = u + (U - Δt Ω) / H - ū

with `ū` the depth mean the vertical solver received and `Ω` the column-integrated boundary stress the substepping
carried. Subtracting `Δt Ω` counts the boundary flux once, as the solver realized it. Without an implicit boundary
flux `Ω` and `ū` are absent and this reduces to `u = u + (U - ∫u dz) / H`.
"""
function barotropic_split_explicit_corrector!(u, v, free_surface, grid, Δt)
    state = free_surface.filtered_state
    U, V  = free_surface.barotropic_velocities
    U̅, V̅  = state.U̅, state.V̅
    arch  = architecture(grid)

    # Preparing velocities for the barotropic correction
    mask_immersed_field!(u)
    mask_immersed_field!(v)

    cᵁ, cⱽ = barotropic_boundary_coefficients(free_surface.implicit_boundary_coefficients)
    reconcile_barotropic_mode!(u, v, U, V, U̅, V̅, cᵁ, cⱽ, Δt, grid, arch)

    return nothing
end

function reconcile_barotropic_mode!(u, v, U, V, U̅, V̅, ::Nothing, ::Nothing, Δt, grid, arch)
    compute_barotropic_mode!(U̅, V̅, grid, u, v)
    launch!(arch, grid, volume_kernel_parameters(grid), _barotropic_split_explicit_corrector!,
            u, v, U, V, U̅, V̅, grid)
    return nothing
end

# `U̅` and `V̅` hold the pre-solve depth means, so the correction must not recompute them here.
function reconcile_barotropic_mode!(u, v, U, V, U̅, V̅, cᵁ, cⱽ, Δt, grid, arch)
    launch!(arch, grid, volume_kernel_parameters(grid), _correct_barotropic_mode_with_boundary_flux!,
            u, v, U, V, U̅, V̅, cᵁ, cⱽ, Δt, grid)

    compute_barotropic_mode!(U, V, grid, u, v)
    return nothing
end

@kernel function _correct_barotropic_mode_with_boundary_flux!(u, v, U, V, U̅, V̅, cᵁ, cⱽ, Δt, grid)
    i, j, k = @index(Global, NTuple)
    Hᶠᶜ = column_depthᶠᶜᵃ(i, j, grid)
    Hᶜᶠ = column_depthᶜᶠᵃ(i, j, grid)

    immersedᶠᶜᶜ = immersed_peripheral_node(i, j, k, grid, Face(), Center(), Center()) | immersed_inactive_node(i, j, k, grid, Face(), Center(), Center())
    immersedᶜᶠᶜ = immersed_peripheral_node(i, j, k, grid, Center(), Face(), Center()) | immersed_inactive_node(i, j, k, grid, Center(), Face(), Center())

    δuᵢ = @inbounds U[i, j, 1] - Δt * barotropic_correction(i, j, grid, cᵁ)
    δvⱼ = @inbounds V[i, j, 1] - Δt * barotropic_correction(i, j, grid, cⱽ)

    u_correction = @inbounds ifelse(Hᶠᶜ == 0, zero(grid), δuᵢ / Hᶠᶜ - U̅[i, j, 1])
    v_correction = @inbounds ifelse(Hᶜᶠ == 0, zero(grid), δvⱼ / Hᶜᶠ - V̅[i, j, 1])

    @inbounds u[i, j, k] = ifelse(immersedᶠᶜᶜ, zero(grid), u[i, j, k] + u_correction)
    @inbounds v[i, j, k] = ifelse(immersedᶜᶠᶜ, zero(grid), v[i, j, k] + v_correction)
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
function compute_transport_velocities!(model, free_surface::SplitExplicitFreeSurface)
    grid = model.grid
    u, v, w = model.velocities
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
