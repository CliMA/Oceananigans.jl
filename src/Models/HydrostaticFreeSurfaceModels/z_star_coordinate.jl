using Oceananigans.Grids: halo_size, topology, AbstractGrid, Flat,
    column_depthᶜᶜᵃ, column_depthᶜᶠᵃ, column_depthᶠᶜᵃ, column_depthᶠᶠᵃ,
    static_column_depthᶜᶜᵃ, static_column_depthᶜᶠᵃ, static_column_depthᶠᶜᵃ, static_column_depthᶠᶠᵃ
using Oceananigans.ImmersedBoundaries: MutableGridOfSomeKind

import Oceananigans: prognostic_state, restore_prognostic_state!

#####
##### Mutable-specific vertical spacings update
#####

# The easy case
barotropic_transport(free_surface::SplitExplicitFreeSurface) =
    (U = free_surface.filtered_state.Ũ,
     V = free_surface.filtered_state.Ṽ)

# The easy case
barotropic_velocities(free_surface::SplitExplicitFreeSurface) =
    free_surface.barotropic_velocities

# The "harder" case, barotropic velocities are computed on the fly
barotropic_velocities(free_surface) = nothing, nothing
barotropic_transport(free_surface)  = nothing, nothing

"""
    ab2_step_grid!(grid::MutableGridOfSomeKind, model, ::ZStarCoordinate, Δt, χ)

Update z-star grid scaling factors during an AB2 time step.

Copies the free surface height `η` from the model to the grid's internal storage,
then recomputes the grid stretching factors `σ` at all staggered locations.
The previous scaling `σᶜᶜ⁻` is also updated for use in tracer evolution.
"""
function ab2_step_grid!(grid::MutableGridOfSomeKind, model, ztype::ZStarCoordinate, Δt, χ)
    parent(grid.z.σᶜᶜ⁻) .= parent(grid.z.σᶜᶜⁿ)
    launch!(architecture(grid), grid, surface_kernel_parameters(grid), _update_zstar_scaling!, model.free_surface.displacement, grid)
    return nothing
end

"""
    rk_substep_grid!(grid::MutableGridOfSomeKind, model, ::ZStarCoordinate, Δt)

Update z-star grid scaling factors during a split Runge-Kutta substep.

Similar to `ab2_step_grid!`, but only updates `σᶜᶜ⁻` on the final substep
(when `model.clock.stage == model.timestepper.Nstages`).
"""
function rk_substep_grid!(grid::MutableGridOfSomeKind, model, ztype::ZStarCoordinate, Δt)
    parent(grid.z.σᶜᶜ⁻) .= parent(grid.z.σᶜᶜⁿ)
    launch!(architecture(grid), grid, surface_kernel_parameters(grid), _update_zstar_scaling!, model.free_surface.displacement, grid)
    return nothing
end

# Update η in the grid
@kernel function _update_zstar_scaling!(ηⁿ⁺¹, grid)
    i, j = @index(Global, NTuple)
    @inbounds grid.z.ηⁿ[i, j, 1] = ηⁿ⁺¹[i, j, grid.Nz+1]
    update_grid_scaling!(grid.z, i, j, grid)
end

@inline function update_grid_scaling!(z_coordinate, i, j, grid)
    hᶜᶜ = static_column_depthᶜᶜᵃ(i, j, grid)
    hᶠᶜ = static_column_depthᶠᶜᵃ(i, j, grid)
    hᶜᶠ = static_column_depthᶜᶠᵃ(i, j, grid)
    hᶠᶠ = static_column_depthᶠᶠᵃ(i, j, grid)

    Hᶜᶜ = column_depthᶜᶜᵃ(i, j, 1, grid, z_coordinate.ηⁿ)
    Hᶠᶜ = column_depthᶠᶜᵃ(i, j, 1, grid, z_coordinate.ηⁿ)
    Hᶜᶠ = column_depthᶜᶠᵃ(i, j, 1, grid, z_coordinate.ηⁿ)
    Hᶠᶠ = column_depthᶠᶠᵃ(i, j, 1, grid, z_coordinate.ηⁿ)

    σᶜᶜ = ifelse(hᶜᶜ == 0, one(grid), Hᶜᶜ / hᶜᶜ)
    σᶠᶜ = ifelse(hᶠᶜ == 0, one(grid), Hᶠᶜ / hᶠᶜ)
    σᶜᶠ = ifelse(hᶜᶠ == 0, one(grid), Hᶜᶠ / hᶜᶠ)
    σᶠᶠ = ifelse(hᶠᶠ == 0, one(grid), Hᶠᶠ / hᶠᶠ)

    @inbounds begin
        # update current scaling
        z_coordinate.σᶜᶜⁿ[i, j, 1] = σᶜᶜ
        z_coordinate.σᶠᶜⁿ[i, j, 1] = σᶠᶜ
        z_coordinate.σᶜᶠⁿ[i, j, 1] = σᶜᶠ
        z_coordinate.σᶠᶠⁿ[i, j, 1] = σᶠᶠ
    end
end

"""
    update_grid_vertical_velocity!(velocities, model, grid::MutableGridOfSomeKind, ::ZStarCoordinate; parameters)

Compute the time derivative of the z-star grid stretching factor `∂t_σ`.

For z-star coordinates, `∂t_σ = -∇·U / H` where `U` is the barotropic transport
and `H` is the static column depth. This represents the rate of change of the
vertical grid spacing due to free surface motion.

The barotropic transport is obtained from `barotropic_velocities` for prognostic
velocities or `barotropic_transport` for transport velocities (which may differ
when using split-explicit free surface).
"""
function update_grid_vertical_velocity!(velocities, model, grid::MutableGridOfSomeKind, ::ZStarCoordinate; parameters=surface_kernel_parameters(grid))

    # the barotropic velocities are retrieved from the free surface model for a
    # SplitExplicitFreeSurface and are calculated for other free surface models
    # Here we distinguish between the model (prognostic) velocities and the transport velocities
    # used to advect tracers...
    if velocities === model.velocities
        U, V = barotropic_velocities(model.free_surface)
    else
        U, V = barotropic_transport(model.free_surface)
    end

    u, v, _ = velocities
    ∂t_σ    = grid.z.∂t_σ

    # Update the time derivative of the vertical spacing,
    # No need to fill the halo as the scaling is updated _IN_ the halos through the parameters
    launch!(architecture(grid), grid, parameters, _update_grid_vertical_velocity!, ∂t_σ, grid, U, V, u, v)

    return nothing
end

@kernel function _update_grid_vertical_velocity!(∂t_σ, grid, U, V, u, v)
    i, j = @index(Global, NTuple)
    kᴺ = size(grid, 3)

    hᶜᶜ = static_column_depthᶜᶜᵃ(i, j, grid)

    # ∂(η / H)/∂t = - ∇ ⋅ ∫udz / H
    δx_U = δxᶜᶜᶜ(i, j, kᴺ, grid, Δy_qᶠᶜᶜ, barotropic_U, U, u)
    δy_V = δyᶜᶜᶜ(i, j, kᴺ, grid, Δx_qᶜᶠᶜ, barotropic_V, V, v)

    δh_U = (δx_U + δy_V) * Az⁻¹ᶜᶜᶜ(i, j, kᴺ, grid)

    @inbounds ∂t_σ[i, j, 1] = ifelse(hᶜᶜ == 0, zero(grid), - δh_U / hᶜᶜ)
end

#####
##### Multiply by grid scaling
#####

# fallback
scale_by_stretching_factor!(Gⁿ, tracers, grid) = nothing

"""
    scale_by_stretching_factor!(Gⁿ, tracers, grid::MutableGridOfSomeKind)

Multiply tracer tendencies by the grid stretching factor `σ` for z-star coordinates.

For z-star coordinates, the evolved quantity is `σ * c` rather than `c` alone.
This function scales tendencies after they are computed so that the time-stepping
advances `σ * c` correctly.
"""
function scale_by_stretching_factor!(Gⁿ, tracers, grid::MutableGridOfSomeKind)

    # Multiply the Gⁿ tendencies by the grid scaling
    for i in propertynames(tracers)
        @inbounds G = Gⁿ[i]
        launch!(architecture(grid), grid, :xyz, _scale_by_stretching_factor!, G, grid)
    end

    return nothing
end

@kernel function _scale_by_stretching_factor!(G, grid)
    i, j, k = @index(Global, NTuple)
    @inbounds G[i, j, k] *= σⁿ(i, j, k, grid, Center(), Center(), Center())
end

#####
##### Reconcile vertical coordinate
#####

"""
    reconcile_vertical_coordinate!(vertical_coordinate, model, grid)

Reconcile the vertical coordinate with the current free surface displacement.

For `ZCoordinate` (static grids), this is a no-op.
For `ZStarCoordinate`, recomputes the grid stretching factors `σ` from the
free surface height (we assume that `∂t_σ = 0`).
"""
reconcile_vertical_coordinate!(::ZCoordinate, model, grid) = nothing

function reconcile_vertical_coordinate!(::ZStarCoordinate, model, grid::MutableGridOfSomeKind)
    launch!(architecture(grid), grid, surface_kernel_parameters(grid), _update_zstar_scaling!, model.free_surface.displacement, grid)
    parent(grid.z.σᶜᶜ⁻) .= parent(grid.z.σᶜᶜⁿ)
    return nothing
end

#####
##### Checkpointing
#####

prognostic_state(::ZCoordinate, grid) = nothing
restore_prognostic_state!(::ZCoordinate, grid, ::Nothing) = ZCoordinate()

function prognostic_state(::ZStarCoordinate, grid)
    z = grid.z
    return (ηⁿ   = prognostic_state(z.ηⁿ),
            σᶜᶜⁿ = prognostic_state(z.σᶜᶜⁿ),
            σᶠᶜⁿ = prognostic_state(z.σᶠᶜⁿ),
            σᶜᶠⁿ = prognostic_state(z.σᶜᶠⁿ),
            σᶠᶠⁿ = prognostic_state(z.σᶠᶠⁿ),
            σᶜᶜ⁻ = prognostic_state(z.σᶜᶜ⁻))
end

function restore_prognostic_state!(::ZStarCoordinate, grid, from)
    z = grid.z
    restore_prognostic_state!(z.ηⁿ,   from.ηⁿ)
    restore_prognostic_state!(z.σᶜᶜⁿ, from.σᶜᶜⁿ)
    restore_prognostic_state!(z.σᶠᶜⁿ, from.σᶠᶜⁿ)
    restore_prognostic_state!(z.σᶜᶠⁿ, from.σᶜᶠⁿ)
    restore_prognostic_state!(z.σᶠᶠⁿ, from.σᶠᶠⁿ)
    restore_prognostic_state!(z.σᶜᶜ⁻, from.σᶜᶜ⁻)
    return ZStarCoordinate()
end
