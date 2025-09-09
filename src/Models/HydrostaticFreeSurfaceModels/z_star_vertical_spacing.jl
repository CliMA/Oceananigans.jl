using Oceananigans.Grids
using Oceananigans.Grids: halo_size, topology, AbstractGrid
using Oceananigans.ImmersedBoundaries: MutableGridOfSomeKind

#####
##### Mutable-specific vertical spacings update
#####

# The easy case
barotropic_transport(free_surface::SplitExplicitFreeSurface) = 
    (U = free_surface.filtered_state.Ũ, 
     V = free_surface.filtered_state.Ṽ)

# The easy case
barotropic_velocities(free_surface::SplitExplicitFreeSurface) = free_surface.barotropic_velocities

# The "harder" case, barotropic velocities are computed on the fly
barotropic_velocities(free_surface) = nothing, nothing
barotropic_transport(free_surface)  = nothing, nothing

# Fallback
ab2_step_grid!(grid, model, ztype, Δt, χ) = nothing

function ab2_step_grid!(grid::MutableGridOfSomeKind, model, ztype::ZStarCoordinate, Δt, χ)

    U, V = barotropic_transport(model.free_surface)
    Gⁿ   = ztype.storage
   
    u, v, _ = model.velocities
    
    launch!(architecture(grid), grid, w_kernel_parameters(grid), _ab2_update_grid_scaling!, 
            Gⁿ, grid, Δt, χ, U, V, u, v)

    return nothing
end

# Update η in the grid
# Note!!! This η is different than the free surface coming from the barotropic step!!
# This η is the one used to compute the vertical spacing.
# TODO: The two different free surfaces need to be reconciled.
@kernel function _ab2_update_grid_scaling!(Gⁿ, grid, Δt, χ, U, V, u, v)
    i, j = @index(Global, NTuple)
    kᴺ = size(grid, 3)
    ηⁿ = grid.z.ηⁿ

    C₁ = 3 * one(χ) / 2 + χ
    C₂ =     one(χ) / 2 + χ

    δx_U = δxᶜᶜᶜ(i, j, kᴺ, grid, Δy_qᶠᶜᶜ, barotropic_U, U, u)
    δy_V = δyᶜᶜᶜ(i, j, kᴺ, grid, Δx_qᶜᶠᶜ, barotropic_V, V, v)
    δh_U = (δx_U + δy_V) * Az⁻¹ᶜᶜᶜ(i, j, kᴺ, grid)

    @inbounds ηⁿ[i, j, 1] -= Δt * (C₁ * δh_U - C₂ * Gⁿ[i, j, 1])
    @inbounds Gⁿ[i, j, 1] = δh_U

    update_grid_scaling!(grid.z, i, j, grid)
end

rk3_substep_grid!(grid, model, vertical_coordinate, Δt) = nothing
rk3_substep_grid!(grid::MutableGridOfSomeKind, model, ztype::ZStarCoordinate, Δt) = 
    launch!(architecture(grid), grid, w_kernel_parameters(grid), _rk3_update_grid_scaling!, model.free_surface.η, grid)

# Update η in the grid
@kernel function _rk3_update_grid_scaling!(ηⁿ⁺¹, grid)
    i, j = @index(Global, NTuple)

    @inbounds grid.z.ηⁿ[i, j, 1] = ηⁿ⁺¹[i, j, grid.Nz+1]
    update_grid_scaling!(grid.z, i, j, grid)
end

@inline function update_grid_scaling!(z, i, j, grid)
    hᶜᶜ = static_column_depthᶜᶜᵃ(i, j, grid)
    hᶠᶜ = static_column_depthᶠᶜᵃ(i, j, grid)
    hᶜᶠ = static_column_depthᶜᶠᵃ(i, j, grid)
    hᶠᶠ = static_column_depthᶠᶠᵃ(i, j, grid)

    Hᶜᶜ = column_depthᶜᶜᵃ(i, j, 1, grid, z.ηⁿ)
    Hᶠᶜ = column_depthᶠᶜᵃ(i, j, 1, grid, z.ηⁿ)
    Hᶜᶠ = column_depthᶜᶠᵃ(i, j, 1, grid, z.ηⁿ)
    Hᶠᶠ = column_depthᶠᶠᵃ(i, j, 1, grid, z.ηⁿ)

    σᶜᶜ = ifelse(hᶜᶜ == 0, one(grid), Hᶜᶜ / hᶜᶜ)
    σᶠᶜ = ifelse(hᶠᶜ == 0, one(grid), Hᶠᶜ / hᶠᶜ)
    σᶜᶠ = ifelse(hᶜᶠ == 0, one(grid), Hᶜᶠ / hᶜᶠ)
    σᶠᶠ = ifelse(hᶠᶠ == 0, one(grid), Hᶠᶠ / hᶠᶠ)

    @inbounds begin
        # Update previous scaling
        z.σᶜᶜ⁻[i, j, 1] = z.σᶜᶜⁿ[i, j, 1]

        # update current scaling
        z.σᶜᶜⁿ[i, j, 1] = σᶜᶜ
        z.σᶠᶜⁿ[i, j, 1] = σᶠᶜ
        z.σᶜᶠⁿ[i, j, 1] = σᶜᶠ
        z.σᶠᶠⁿ[i, j, 1] = σᶠᶠ
    end
end

update_grid_vertical_velocity!(model, grid, ztype; kw...) = nothing

function update_grid_vertical_velocity!(velocities, grid::MutableGridOfSomeKind, ::ZStarCoordinate; parameters = w_kernel_parameters(grid))

    # the barotropic velocities are retrieved from the free surface model for a
    # SplitExplicitFreeSurface and are calculated for other free surface models
    U, V    = nothing, nothing #barotropic_velocities(model.free_surface)
    u, v, _ = velocities
    ∂t_σ    = grid.z.∂t_σ

    # Update the time derivative of the vertical spacing,
    # No need to fill the halo as the scaling is updated _IN_ the halos
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
##### ZStarCoordinate-specific implementation of the additional terms to be included in the momentum equations
#####

# Fallbacks
@inline grid_slope_contribution_x(i, j, k, grid, buoyancy, ztype, model_fields) = zero(grid)
@inline grid_slope_contribution_y(i, j, k, grid, buoyancy, ztype, model_fields) = zero(grid)

@inline grid_slope_contribution_x(i, j, k, grid::MutableGridOfSomeKind, ::Nothing, ::ZStarCoordinate, model_fields) = zero(grid)
@inline grid_slope_contribution_y(i, j, k, grid::MutableGridOfSomeKind, ::Nothing, ::ZStarCoordinate, model_fields) = zero(grid)

@inline ∂x_z(i, j, k, grid) = ∂xᶠᶜᶜ(i, j, k, grid, znode, Center(), Center(), Center())
@inline ∂y_z(i, j, k, grid) = ∂yᶜᶠᶜ(i, j, k, grid, znode, Center(), Center(), Center())

@inline grid_slope_contribution_x(i, j, k, grid::MutableGridOfSomeKind, buoyancy, ::ZStarCoordinate, model_fields) =
    ℑxᶠᵃᵃ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, buoyancy.formulation, model_fields) * ∂x_z(i, j, k, grid)

@inline grid_slope_contribution_y(i, j, k, grid::MutableGridOfSomeKind, buoyancy, ::ZStarCoordinate, model_fields) =
    ℑyᵃᶠᵃ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, buoyancy.formulation, model_fields) * ∂y_z(i, j, k, grid)
