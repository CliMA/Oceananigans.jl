using Oceananigans.Grids

using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces: dynamic_column_depthᶜᶜᵃ,
                                                                                  dynamic_column_depthᶜᶠᵃ,
                                                                                  dynamic_column_depthᶠᶜᵃ,
                                                                                  dynamic_column_depthᶠᶠᵃ

#####
##### ZStar-specific vertical spacings update
#####

function update_grid!(model, grid::ZStarSpacingGrid; parameters = :xy)

    # Scaling (just update once, they are the same for all the metrics)
    sᶜᶜ⁻   = grid.Δzᵃᵃᶠ.sᶜᶜ⁻
    sᶜᶜⁿ   = grid.Δzᵃᵃᶠ.sᶜᶜⁿ
    sᶠᶜ⁻   = grid.Δzᵃᵃᶠ.sᶠᶜ⁻
    sᶠᶜⁿ   = grid.Δzᵃᵃᶠ.sᶠᶜⁿ
    sᶜᶠ⁻   = grid.Δzᵃᵃᶠ.sᶜᶠ⁻
    sᶜᶠⁿ   = grid.Δzᵃᵃᶠ.sᶜᶠⁿ
    sᶠᶠⁿ   = grid.Δzᵃᵃᶠ.sᶠᶠⁿ
    ∂t_s   = grid.Δzᵃᵃᶠ.∂t_s
    η_grid = grid.zᵃᵃᶠ.∂t_s

    # Free surface variables:
    # TODO: At the moment only SplitExplicitFreeSurface is supported,
    # but zstar can be extended to other free surface solvers by calculating
    # the barotropic velocity in this step
    U = model.free_surface.barotropic_velocities.U 
    V = model.free_surface.barotropic_velocities.V 
    η = model.free_surface.η

    # Update vertical spacing with available parameters 
    # No need to fill the halo as the scaling is updated _IN_ the halos
    launch!(architecture(grid), grid, parameters, _update_zstar!, 
            sᶜᶜⁿ, sᶠᶜⁿ, sᶜᶠⁿ, sᶠᶠⁿ, sᶜᶜ⁻, sᶠᶜ⁻, sᶜᶠ⁻, η_grid, η, grid)

    # Update the time derivative of the grid-scaling. Note that in this case we leverage the
    # free surface evolution equation, where the time derivative of the free surface is equal
    # to the divergence of the vertically integrated velocity field, such that
    # ∂ₜ((H + η) / H) = H⁻¹ ∂ₜη =  - H⁻¹ ∇ ⋅ ∫udz 
    launch!(architecture(grid), grid, parameters, _update_∂t_s!, 
            ∂t_s, U, V, grid)

    return nothing
end

# NOTE: The ZStar vertical spacing only supports a SplitExplicitFreeSurface
# TODO: extend to support other free surface solvers
@kernel function _update_∂t_s!(∂t_s, U, V, grid)
    i, j  = @index(Global, NTuple)
    kᴺ = size(grid, 3)

    # ∂(η / H)/∂t = - ∇ ⋅ ∫udz / H
    δx_U = δxᶜᶜᶜ(i, j, kᴺ, grid, Δy_qᶠᶜᶜ, U)
    δy_V = δyᶜᶜᶜ(i, j, kᴺ, grid, Δx_qᶜᶠᶜ, V)

    δh_U = (δx_U + δy_V) / Azᶜᶜᶜ(i, j, kᴺ, grid)
    H    = static_column_depthᶜᶜᵃ(i, j, grid)

    @inbounds ∂t_s[i, j] = ifelse(H == 0, zero(grid), - δh_U / H)
end

@kernel function _update_zstar!(sᶜᶜⁿ, sᶠᶜⁿ, sᶜᶠⁿ, sᶠᶠⁿ, sᶜᶜ⁻, sᶠᶜ⁻, sᶜᶠ⁻, η_grid, η, grid)
    i, j = @index(Global, NTuple)
    k_top = grid.Nz+1

    hᶜᶜ = static_column_depthᶜᶜᵃ(i, j, grid)
    hᶠᶜ = static_column_depthᶠᶜᵃ(i, j, grid)
    hᶜᶠ = static_column_depthᶜᶠᵃ(i, j, grid)
    hᶠᶠ = static_column_depthᶠᶠᵃ(i, j, grid)

    Hᶜᶜ = dynamic_column_depthᶜᶜᵃ(i, j, grid, η)
    Hᶠᶜ = dynamic_column_depthᶠᶜᵃ(i, j, grid, η)
    Hᶜᶠ = dynamic_column_depthᶜᶠᵃ(i, j, grid, η)
    Hᶠᶠ = dynamic_column_depthᶠᶠᵃ(i, j, grid, η)

    @inbounds begin
        sᶜᶜ = ifelse(hᶜᶜ == 0, one(grid), Hᶜᶜ / hᶜᶜ)
        sᶠᶜ = ifelse(hᶠᶜ == 0, one(grid), Hᶠᶜ / hᶠᶜ)
        sᶜᶠ = ifelse(hᶜᶠ == 0, one(grid), Hᶜᶠ / hᶜᶠ)
        sᶠᶠ = ifelse(hᶠᶠ == 0, one(grid), Hᶠᶠ / hᶠᶠ)

        # Update previous scaling
        sᶜᶜ⁻[i, j] = sᶜᶜⁿ[i, j]
        sᶠᶜ⁻[i, j] = sᶠᶜⁿ[i, j]
        sᶜᶠ⁻[i, j] = sᶜᶠⁿ[i, j]
        
        # update current scaling
        sᶜᶜⁿ[i, j] = sᶜᶜ
        sᶠᶜⁿ[i, j] = sᶠᶜ
        sᶜᶠⁿ[i, j] = sᶜᶠ
        sᶠᶠⁿ[i, j] = sᶠᶠ

        # Update η in the grid
        η_grid[i, j] = η[i, j, k_top]
    end
end

#####
##### ZStar-specific implementation of the additional terms to be included in the momentum equations
#####

@inline z_minus_rᶜᶜᶜ(i, j, k, grid, η) = @inbounds η[i, j, grid.Nz+1] * (1 + rnode(i, j, k, grid, Center(), Center(), Center()) / static_column_depthᶜᶜᵃ(i, j, grid))

@inline ∂x_z(i, j, k, grid, free_surface) = @inbounds ∂xᶠᶜᶜ(i, j, k, grid, z_minus_rᶜᶜᶜ, free_surface.η)
@inline ∂y_z(i, j, k, grid, free_surface) = @inbounds ∂yᶜᶠᶜ(i, j, k, grid, z_minus_rᶜᶜᶜ, free_surface.η)

@inline grid_slope_contribution_x(i, j, k, grid::ZStarSpacingGrid, free_surface, ::Nothing, model_fields) = zero(grid)
@inline grid_slope_contribution_y(i, j, k, grid::ZStarSpacingGrid, free_surface, ::Nothing, model_fields) = zero(grid)

@inline grid_slope_contribution_x(i, j, k, grid::ZStarSpacingGrid, free_surface, buoyancy, model_fields) = 
    ℑxᶠᵃᵃ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, buoyancy.model, model_fields) * ∂x_z(i, j, k, grid, free_surface)

@inline grid_slope_contribution_y(i, j, k, grid::ZStarSpacingGrid, free_surface, buoyancy, model_fields) = 
    ℑyᵃᶠᵃ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, buoyancy.model, model_fields) * ∂y_z(i, j, k, grid, free_surface)
