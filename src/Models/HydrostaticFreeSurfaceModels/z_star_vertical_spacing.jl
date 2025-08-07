using Oceananigans.Grids
using Oceananigans.Grids: halo_size, topology, AbstractGrid
using Oceananigans.ImmersedBoundaries: MutableGridOfSomeKind

#####
##### Mutable-specific vertical spacings update
#####

# The easy case
barotropic_velocities(free_surface::SplitExplicitFreeSurface) = free_surface.barotropic_velocities

# The "harder" case, barotropic velocities are computed on the fly
barotropic_velocities(free_surface) = nothing, nothing

# Fallback
ab2_step_grid!(grid, model, ztype, Δt, χ) = nothing

function zstar_params(grid::AbstractGrid)

    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)
    Tx, Ty, _ = topology(grid)

    xrange = params_range(Hx, Nx, Tx)
    yrange = params_range(Hy, Ny, Ty)

    return KernelParameters(xrange, yrange)
end

params_range(H, N, ::Type{Flat}) = 1:1
params_range(H, N, T) = -H+2:N+H-1

function ab2_step_grid!(grid::MutableGridOfSomeKind, model, ::ZStar, Δt, χ)

    # Scalings and free surface
    σᶜᶜ⁻ = grid.z.σᶜᶜ⁻
    σᶜᶜⁿ = grid.z.σᶜᶜⁿ
    σᶠᶜⁿ = grid.z.σᶠᶜⁿ
    σᶜᶠⁿ = grid.z.σᶜᶠⁿ
    σᶠᶠⁿ = grid.z.σᶠᶠⁿ
    ηⁿ   = grid.z.ηⁿ
    Gⁿ   = grid.z.Gⁿ
  
    U, V = barotropic_velocities(model.free_surface)
    u, v, _ = model.velocities

    params = zstar_params(grid)

    launch!(architecture(grid), grid, params, _ab2_update_grid_scaling!,
            σᶜᶜⁿ, σᶠᶜⁿ, σᶜᶠⁿ, σᶠᶠⁿ, σ⁻¹ᶜᶜⁿ, σ⁻¹ᶠᶜⁿ, σ⁻¹ᶜᶠⁿ, σ⁻¹ᶠᶠⁿ, σᶜᶜ⁻, ηⁿ, Gⁿ, grid, Δt, χ, U, V, u, v)

    return nothing
end

# Update η in the grid
# Note!!! This η is different than the free surface coming from the barotropic step!!
# This η is the one used to compute the vertical spacing.
# TODO: The two different free surfaces need to be reconciled.
@kernel function _ab2_update_grid_scaling!(σᶜᶜⁿ, σᶠᶜⁿ, σᶜᶠⁿ, σᶠᶠⁿ, σ⁻¹ᶜᶜⁿ, σ⁻¹ᶠᶜⁿ, σ⁻¹ᶜᶠⁿ, σ⁻¹ᶠᶠⁿ, σᶜᶜ⁻, ηⁿ, Gⁿ, grid, Δt, χ, U, V, u, v)
    i, j = @index(Global, NTuple)
    kᴺ = size(grid, 3)

    C₁ = 3 * one(χ) / 2 + χ
    C₂ =     one(χ) / 2 + χ

    δx_U = δxᶜᶜᶜ(i, j, kᴺ, grid, Δy_qᶠᶜᶜ, barotropic_U, U, u)
    δy_V = δyᶜᶜᶜ(i, j, kᴺ, grid, Δx_qᶜᶠᶜ, barotropic_V, V, v)
    δh_U = (δx_U + δy_V) * Az⁻¹ᶜᶜᶜ(i, j, kᴺ, grid)

    @inbounds ηⁿ[i, j, 1] -= Δt * (C₁ * δh_U - C₂ * Gⁿ[i, j, 1])
    @inbounds Gⁿ[i, j, 1] = δh_U

    update_grid_scaling!(σᶜᶜⁿ, σᶠᶜⁿ, σᶜᶠⁿ, σᶠᶠⁿ, σᶜᶜ⁻, i, j, grid, ηⁿ)
end

rk3_substep_grid!(grid, model, vertical_coordinate, Δt, γⁿ, ζⁿ) = nothing
rk3_substep_grid!(grid::MutableGridOfSomeKind, model, ztype::ZStar, Δt, ::Nothing, ::Nothing) =
    rk3_substep_grid!(grid, model, ztype, Δt, one(grid), zero(grid))

function rk3_substep_grid!(grid::MutableGridOfSomeKind, model, ::ZStar, Δt, γⁿ, ζⁿ)

    # Scalings and free surface
    σᶜᶜ⁻ = grid.z.σᶜᶜ⁻
    σᶜᶜⁿ = grid.z.σᶜᶜⁿ
    σᶠᶜⁿ = grid.z.σᶠᶜⁿ
    σᶜᶠⁿ = grid.z.σᶜᶠⁿ
    σᶠᶠⁿ = grid.z.σᶠᶠⁿ
    ηⁿ   = grid.z.ηⁿ
    ηⁿ⁻¹ = grid.z.Gⁿ

    U, V = barotropic_velocities(model.free_surface)
    u, v, _ = model.velocities
    params = zstar_params(grid)

    launch!(architecture(grid), grid, params, _rk3_update_grid_scaling!,
            σᶜᶜⁿ, σᶠᶜⁿ, σᶜᶠⁿ, σᶠᶠⁿ, σᶜᶜ⁻, ηⁿ, ηⁿ⁻¹, grid, Δt, γⁿ, ζⁿ, U, V, u, v)

    return nothing
end

# Update η in the grid
# Note!!! This η is different than the free surface coming from the barotropic step!!
# This η is the one used to compute the vertical spacing.
# TODO: The two different free surfaces need to be reconciled.
@kernel function _rk3_update_grid_scaling!(σᶜᶜⁿ, σᶠᶜⁿ, σᶜᶠⁿ, σᶠᶠⁿ, σᶜᶜ⁻, ηⁿ, ηⁿ⁻¹, grid, Δt, γⁿ, ζⁿ, U, V, u, v)
    i, j = @index(Global, NTuple)
    kᴺ = size(grid, 3)

    δx_U = δxᶜᶜᶜ(i, j, kᴺ, grid, Δy_qᶠᶜᶜ, barotropic_U, U, u)
    δy_V = δyᶜᶜᶜ(i, j, kᴺ, grid, Δx_qᶜᶠᶜ, barotropic_V, V, v)
    δh_U = (δx_U + δy_V) * Az⁻¹ᶜᶜᶜ(i, j, kᴺ, grid)

    @inbounds ηⁿ[i, j, 1] = ζⁿ * ηⁿ⁻¹[i, j, 1] + γⁿ * (ηⁿ[i, j, 1] - Δt * δh_U)

    update_grid_scaling!(σᶜᶜⁿ, σᶠᶜⁿ, σᶜᶠⁿ, σᶠᶠⁿ, σᶜᶜ⁻, i, j, grid, ηⁿ)
end

@inline function update_grid_scaling!(σᶜᶜⁿ, σᶠᶜⁿ, σᶜᶠⁿ, σᶠᶠⁿ, σᶜᶜ⁻, i, j, grid, ηⁿ)
    hᶜᶜ = static_column_depthᶜᶜᵃ(i, j, grid)
    hᶠᶜ = static_column_depthᶠᶜᵃ(i, j, grid)
    hᶜᶠ = static_column_depthᶜᶠᵃ(i, j, grid)
    hᶠᶠ = static_column_depthᶠᶠᵃ(i, j, grid)

    Hᶜᶜ = column_depthᶜᶜᵃ(i, j, 1, grid, ηⁿ)
    Hᶠᶜ = column_depthᶠᶜᵃ(i, j, 1, grid, ηⁿ)
    Hᶜᶠ = column_depthᶜᶠᵃ(i, j, 1, grid, ηⁿ)
    Hᶠᶠ = column_depthᶠᶠᵃ(i, j, 1, grid, ηⁿ)

    σᶜᶜ = ifelse(hᶜᶜ == 0, one(grid), Hᶜᶜ / hᶜᶜ)
    σᶠᶜ = ifelse(hᶠᶜ == 0, one(grid), Hᶠᶜ / hᶠᶜ)
    σᶜᶠ = ifelse(hᶜᶠ == 0, one(grid), Hᶜᶠ / hᶜᶠ)
    σᶠᶠ = ifelse(hᶠᶠ == 0, one(grid), Hᶠᶠ / hᶠᶠ)

    @inbounds begin
        # Update previous scaling
        σᶜᶜ⁻[i, j, 1] = σᶜᶜⁿ[i, j, 1]

        # update current scaling
        σᶜᶜⁿ[i, j, 1] = σᶜᶜ
        σᶠᶜⁿ[i, j, 1] = σᶠᶜ
        σᶜᶠⁿ[i, j, 1] = σᶜᶠ
        σᶠᶠⁿ[i, j, 1] = σᶠᶠ

        # Update η in the grid
        ηⁿ[i, j, 1] = η[i, j, k_top]
    end
end

update_grid_vertical_velocity!(model, grid, ztype) = nothing

function update_grid_vertical_velocity!(model, grid::MutableGridOfSomeKind, ::ZStar)

    # the barotropic velocities are retrieved from the free surface model for a
    # SplitExplicitFreeSurface and are calculated for other free surface models
    U, V = barotropic_velocities(model.free_surface)
    u, v, _ = model.velocities
    ∂t_σ  = grid.z.∂t_σ

    params = zstar_params(grid)

    # Update the time derivative of the vertical spacing,
    # No need to fill the halo as the scaling is updated _IN_ the halos
    launch!(architecture(grid), grid, params, _update_grid_vertical_velocity!, ∂t_σ, grid, U, V, u, v)

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

# If U and V exist, we use them
@inline barotropic_U(i, j, k, grid, U, u) = @inbounds U[i, j, k]
@inline barotropic_V(i, j, k, grid, V, v) = @inbounds V[i, j, k]

# If either U or V are not available, we compute them
@inline function barotropic_U(i, j, k, grid, ::Nothing, u)
    U = 0
    for k in 1:size(grid, 3)
        @inbounds U += u[i, j, k] * Δzᶠᶜᶜ(i, j, k, grid)
    end
    return U
end

@inline function barotropic_V(i, j, k, grid, ::Nothing, v)
    V = 0
    for k in 1:size(grid, 3)
        @inbounds V += v[i, j, k] * Δzᶜᶠᶜ(i, j, k, grid)
    end
    return V
end

#####
##### Multiply by grid scaling
#####

# fallback
scale_by_stretching_factor!(Gⁿ, tracers, grid) = nothing

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
##### ZStar-specific implementation of the additional terms to be included in the momentum equations
#####

# Fallbacks
@inline grid_slope_contribution_x(i, j, k, grid, buoyancy, ztype, model_fields) = zero(grid)
@inline grid_slope_contribution_y(i, j, k, grid, buoyancy, ztype, model_fields) = zero(grid)

@inline grid_slope_contribution_x(i, j, k, grid::MutableGridOfSomeKind, ::Nothing, ::ZStar, model_fields) = zero(grid)
@inline grid_slope_contribution_y(i, j, k, grid::MutableGridOfSomeKind, ::Nothing, ::ZStar, model_fields) = zero(grid)

@inline ∂x_z(i, j, k, grid) = ∂xᶠᶜᶜ(i, j, k, grid, znode, Center(), Center(), Center())
@inline ∂y_z(i, j, k, grid) = ∂yᶜᶠᶜ(i, j, k, grid, znode, Center(), Center(), Center())

@inline grid_slope_contribution_x(i, j, k, grid::MutableGridOfSomeKind, buoyancy, ::ZStar, model_fields) =
    ℑxᶠᵃᵃ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, buoyancy.formulation, model_fields) * ∂x_z(i, j, k, grid)

@inline grid_slope_contribution_y(i, j, k, grid::MutableGridOfSomeKind, buoyancy, ::ZStar, model_fields) =
    ℑyᵃᶠᵃ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, buoyancy.formulation, model_fields) * ∂y_z(i, j, k, grid)
