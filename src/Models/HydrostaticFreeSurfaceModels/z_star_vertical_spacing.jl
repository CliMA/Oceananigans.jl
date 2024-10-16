using Oceananigans.Grids: ZStarUnderlyingGrid, rnode
using Oceananigans.ImmersedBoundaries: ImmersedZStarGrid

const ZStarSpacingGrid = Union{ZStarUnderlyingGrid, ImmersedZStarGrid}

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

    # Free surface variables
    U̅   = model.free_surface.state.U̅ 
    V̅   = model.free_surface.state.V̅ 
    η   = model.free_surface.η

    # Update vertical spacing with available parameters 
    # No need to fill the halo as the scaling is updated _IN_ the halos
    launch!(architecture(grid), grid, parameters, _update_zstar!, 
            sᶜᶜⁿ, sᶠᶜⁿ, sᶜᶠⁿ, sᶠᶠⁿ, sᶜᶜ⁻, sᶠᶜ⁻, sᶜᶠ⁻, η_grid, η, grid)

    # Update the time derivative of the grid-scaling. Note that in this case we leverage the
    # free surface evolution equation, where the time derivative of the free surface is equal
    # to the divergence of the vertically integrated velocity field, such that
    # ∂ₜ((H + η) / H) = H⁻¹ ∂ₜη =  - H⁻¹ ∇ ⋅ ∫udz 
    launch!(architecture(grid), grid, parameters, _update_∂t_s!, 
            ∂t_s, U̅, V̅, grid)

    return nothing
end

# NOTE: The ZStar vertical spacing only supports a SplitExplicitFreeSurface
# TODO: extend to support other free surface solvers
@kernel function _update_∂t_s!(∂t_s, U̅, V̅, Hᶜᶜ, grid)
    i, j  = @index(Global, NTuple)
    k_top = grid.Nz + 1 
    Hᶜᶜ = domain_depthᶜᶜᵃ(i, j, grid)

    @inbounds begin
        # ∂(η / H)/∂t = - ∇ ⋅ ∫udz / H
        ∂t_s[i, j, 1] = - 1 / Azᶜᶜᶠ(i, j, k_top-1, grid) * (δxᶜᶜᶠ(i, j, k_top-1, grid, Δy_qᶠᶜᶠ, U̅) +
                                                            δyᶜᶜᶠ(i, j, k_top-1, grid, Δx_qᶜᶠᶠ, V̅)) / Hᶜᶜ
end

@kernel function _update_zstar!(sᶜᶜⁿ, sᶠᶜⁿ, sᶜᶠⁿ, sᶠᶠⁿ, sᶜᶜ⁻, sᶠᶜ⁻, sᶜᶠ⁻, η_grid, η, grid)
    i, j = @index(Global, NTuple)
    k_top = grid.Nz+1

    Hᶜᶜ = domain_depthᶜᶜᵃ(i, j, grid)
    Hᶠᶜ = domain_depthᶠᶜᵃ(i, j, grid)
    Hᶜᶠ = domain_depthᶜᶠᵃ(i, j, grid)
    Hᶠᶠ = domain_depthᶠᶠᵃ(i, j, grid)

    @inbounds begin
        hᶜᶜ = (Hᶜᶜ + η[i, j, k_top]) / Hᶜᶜ
        hᶠᶜ = (Hᶠᶜ +  ℑxᶠᵃᵃ(i, j, k_top, grid, η)) / Hᶠᶜ
        hᶜᶠ = (Hᶜᶠ +  ℑyᵃᶠᵃ(i, j, k_top, grid, η)) / Hᶜᶠ
        hᶠᶠ = (Hᶠᶠ + ℑxyᶠᶠᵃ(i, j, k_top, grid, η)) / Hᶠᶠ

        sᶜᶜ⁻[i, j] = sᶜᶜⁿ[i, j]
        sᶠᶜ⁻[i, j] = sᶠᶜⁿ[i, j]
        sᶜᶠ⁻[i, j] = sᶜᶠⁿ[i, j]
        
        # update current and previous scaling
        sᶜᶜⁿ[i, j] = hᶜᶜ
        sᶠᶜⁿ[i, j] = hᶠᶜ
        sᶜᶠⁿ[i, j] = hᶜᶠ
        sᶠᶠⁿ[i, j] = hᶠᶠ

        # Update η in the grid
        η_grid[i, j] = η[i, j, k_top]
    end
end

#####
##### ZStar-specific implementation of the additional terms to be included in the momentum equations
#####

@inline z_minus_rᶜᶜᶜ(i, j, k, grid, η) = @inbounds η[i, j, grid.Nz+1] * (1 + rnode(i, j, k, grid, Center(), Center(), Center()) / domain_depthᶜᶜᵃ(i, j, grid))

@inline ∂x_z(i, j, k, grid, free_surface) = @inbounds ∂xᶠᶜᶜ(i, j, k, grid, z_minus_rᶜᶜᶜ, free_surface.η)
@inline ∂y_z(i, j, k, grid, free_surface) = @inbounds ∂yᶜᶠᶜ(i, j, k, grid, z_minus_rᶜᶜᶜ, free_surface.η)

@inline grid_slope_contribution_x(i, j, k, grid::ZStarSpacingGrid, free_surface, ::Nothing, model_fields) = zero(grid)
@inline grid_slope_contribution_y(i, j, k, grid::ZStarSpacingGrid, free_surface, ::Nothing, model_fields) = zero(grid)

@inline grid_slope_contribution_x(i, j, k, grid::ZStarSpacingGrid, free_surface, buoyancy, model_fields) = 
    ℑxᶠᵃᵃ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, buoyancy.model, model_fields) * ∂x_z(i, j, k, grid, free_surface)

@inline grid_slope_contribution_y(i, j, k, grid::ZStarSpacingGrid, free_surface, buoyancy, model_fields) = 
    ℑyᵃᶠᵃ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, buoyancy.model, model_fields) * ∂y_z(i, j, k, grid, free_surface)
