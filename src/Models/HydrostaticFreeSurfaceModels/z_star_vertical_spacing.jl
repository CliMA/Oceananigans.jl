using Oceananigans.Grids
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

    # Free surface variables:
    # TODO: At the moment only SplitExplicitFreeSurface is supported,
    # but zstar can be extended to other free surface solvers by calculating
    # the barotropic velocity in this step
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
@kernel function _update_∂t_s!(∂t_s, U̅, V̅, grid)
    i, j  = @index(Global, NTuple)
    k_top = grid.Nz + 1 

    # ∂(η / H)/∂t = - ∇ ⋅ ∫udz / H
    δx_U = ∂xᶠᶜᶜ(i, j, k_top-1, grid, U̅)
    δy_V = ∂yᶜᶠᶜ(i, j, k_top-1, grid, V̅)

    δ_U̅h = (δx_U + δy_V) / Azᶜᶜᶠ(i, j, k_top-1, grid)
    H    = domain_depthᶜᶜᵃ(i, j, grid)

    @show  δ_U̅h, H

    @inbounds  ∂t_s[i, j] = - δ_U̅h / H
end

@kernel function _update_zstar!(sᶜᶜⁿ, sᶠᶜⁿ, sᶜᶠⁿ, sᶠᶠⁿ, sᶜᶜ⁻, sᶠᶜ⁻, sᶜᶠ⁻, η_grid, η, grid)
    i, j = @index(Global, NTuple)
    k_top = grid.Nz+1

    hᶜᶜ = domain_depthᶜᶜᵃ(i, j, grid)
    hᶠᶜ = domain_depthᶠᶜᵃ(i, j, grid)
    hᶜᶠ = domain_depthᶜᶠᵃ(i, j, grid)
    hᶠᶠ = domain_depthᶠᶠᵃ(i, j, grid)

    @inbounds begin
        Hᶜᶜ = (hᶜᶜ +               η[i, j, k_top]) / hᶜᶜ
        Hᶠᶜ = (hᶠᶜ +  ℑxᶠᵃᵃ(i, j, k_top, grid, η)) / hᶠᶜ
        Hᶜᶠ = (hᶜᶠ +  ℑyᵃᶠᵃ(i, j, k_top, grid, η)) / hᶜᶠ
        Hᶠᶠ = (hᶠᶠ + ℑxyᶠᶠᵃ(i, j, k_top, grid, η)) / hᶠᶠ

        sᶜᶜ⁻[i, j] = sᶜᶜⁿ[i, j]
        sᶠᶜ⁻[i, j] = sᶠᶜⁿ[i, j]
        sᶜᶠ⁻[i, j] = sᶜᶠⁿ[i, j]
        
        # update current and previous scaling
        sᶜᶜⁿ[i, j] = Hᶜᶜ
        sᶠᶜⁿ[i, j] = Hᶠᶜ
        sᶜᶠⁿ[i, j] = Hᶜᶠ
        sᶠᶠⁿ[i, j] = Hᶠᶠ

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
