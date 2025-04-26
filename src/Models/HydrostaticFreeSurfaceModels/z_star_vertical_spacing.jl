using Oceananigans.Grids
using Oceananigans.ImmersedBoundaries: MutableGridOfSomeKind

#####
##### Mutable-specific vertical spacings update
#####

# The easy case
barotropic_velocities(free_surface::SplitExplicitFreeSurface) = free_surface.barotropic_velocities

# The "harder" case, barotropic velocities are computed on the fly
barotropic_velocities(free_surface) = nothing, nothing

# Fallback
update_grid!(model, grid, ztype; parameters) = nothing

function update_grid!(model, grid::MutableGridOfSomeKind, ::ZStar; parameters = :xy)

    # Scalings and free surface
    σᶜᶜ⁻  = grid.z.σᶜᶜ⁻
    σᶜᶜⁿ  = grid.z.σᶜᶜⁿ
    σᶠᶜⁿ  = grid.z.σᶠᶜⁿ
    σᶜᶠⁿ  = grid.z.σᶜᶠⁿ
    σᶠᶠⁿ  = grid.z.σᶠᶠⁿ
    ∂t_σ  = grid.z.∂t_σ
    ηⁿ    = grid.z.ηⁿ
    η     = model.free_surface.η

    launch!(architecture(grid), grid, parameters, _update_grid_scaling!,
            σᶜᶜⁿ, σᶠᶜⁿ, σᶜᶠⁿ, σᶠᶠⁿ, σᶜᶜ⁻, ηⁿ, grid, η)

    # the barotropic velocities are retrieved from the free surface model for a
    # SplitExplicitFreeSurface and are calculated for other free surface models
    U, V = barotropic_velocities(model.free_surface)
    u, v, _ = model.velocities

    # Update the time derivative of the vertical spacing,
    # No need to fill the halo as the scaling is updated _IN_ the halos
    launch!(architecture(grid), grid, parameters, _update_grid_vertical_velocity!, ∂t_σ, grid, U, V, u, v)

    return nothing
end

@kernel function _update_grid_scaling!(σᶜᶜⁿ, σᶠᶜⁿ, σᶜᶠⁿ, σᶠᶠⁿ, σᶜᶜ⁻, ηⁿ, grid, η)
    i, j = @index(Global, NTuple)
    k_top = size(grid, 3) + 1

    hᶜᶜ = static_column_depthᶜᶜᵃ(i, j, grid)
    hᶠᶜ = static_column_depthᶠᶜᵃ(i, j, grid)
    hᶜᶠ = static_column_depthᶜᶠᵃ(i, j, grid)
    hᶠᶠ = static_column_depthᶠᶠᵃ(i, j, grid)

    Hᶜᶜ = column_depthᶜᶜᵃ(i, j, k_top, grid, η)
    Hᶠᶜ = column_depthᶠᶜᵃ(i, j, k_top, grid, η)
    Hᶜᶠ = column_depthᶜᶠᵃ(i, j, k_top, grid, η)
    Hᶠᶠ = column_depthᶠᶠᵃ(i, j, k_top, grid, η)

    @inbounds begin
        σᶜᶜ = ifelse(hᶜᶜ == 0, one(grid), Hᶜᶜ / hᶜᶜ)
        σᶠᶜ = ifelse(hᶠᶜ == 0, one(grid), Hᶠᶜ / hᶠᶜ)
        σᶜᶠ = ifelse(hᶜᶠ == 0, one(grid), Hᶜᶠ / hᶜᶠ)
        σᶠᶠ = ifelse(hᶠᶠ == 0, one(grid), Hᶠᶠ / hᶠᶠ)

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

# If U and V exist, we just take them
@inline barotropic_U(i, j, k, grid, U, u) = @inbounds U[i, j, k]
@inline barotropic_V(i, j, k, grid, V, v) = @inbounds V[i, j, k]

# If U and V are not available, we compute them
@inline function barotropic_U(i, j, k, grid, ::Nothing, u)
    U = 0
    for k in 1:size(grid, 3)
        U += u[i, j, k] * Δzᶠᶜᶜ(i, j, k, grid)
    end
    return U
end

@inline function barotropic_V(i, j, k, grid, ::Nothing, v)
    V = 0
    for k in 1:size(grid, 3)
        V += v[i, j, k] * Δzᶜᶠᶜ(i, j, k, grid)
    end
    return V
end

#####
##### ZStar-specific implementation of the additional terms to be included in the momentum equations
#####

# Fallbacks
@inline grid_slope_contribution_x(i, j, k, grid, buoyancy, ztype, model_fields) = zero(grid)
@inline grid_slope_contribution_y(i, j, k, grid, buoyancy, ztype, model_fields) = zero(grid)

@inline grid_slope_contribution_x(i, j, k, grid::MutableGridOfSomeKind, ::Nothing, ::ZStar, model_fields) = zero(grid)
@inline grid_slope_contribution_y(i, j, k, grid::MutableGridOfSomeKind, ::Nothing, ::ZStar, model_fields) = zero(grid)

@inline ∂x_z(i, j, k, grid) = @inbounds ∂xᶠᶜᶜ(i, j, k, grid, znode, Center(), Center(), Center())
@inline ∂y_z(i, j, k, grid) = @inbounds ∂yᶜᶠᶜ(i, j, k, grid, znode, Center(), Center(), Center())

@inline grid_slope_contribution_x(i, j, k, grid::MutableGridOfSomeKind, buoyancy, ::ZStar, model_fields) =
    ℑxᶠᵃᵃ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, buoyancy.formulation, model_fields) * ∂x_z(i, j, k, grid)

@inline grid_slope_contribution_y(i, j, k, grid::MutableGridOfSomeKind, buoyancy, ::ZStar, model_fields) =
    ℑyᵃᶠᵃ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, buoyancy.formulation, model_fields) * ∂y_z(i, j, k, grid)

####
#### Removing the scaling of the vertical coordinate from the tracer fields
####

const EmptyTuples = Union{NamedTuple{(), Tuple{}}, Tuple{}}

unscale_tracers!(::EmptyTuples, ::MutableGridOfSomeKind; kwargs...) = nothing

function unscale_tracers!(tracers, grid::MutableGridOfSomeKind; parameters = :xy)

    for tracer in tracers
        launch!(architecture(grid), grid, parameters, _unscale_tracer!,
                tracer, grid, Val(grid.Hz), Val(grid.Nz))
    end

    return nothing
end

@kernel function _unscale_tracer!(tracer, grid, ::Val{Hz}, ::Val{Nz}) where {Hz, Nz}
    i, j = @index(Global, NTuple)

    @unroll for k in -Hz+1:Nz+Hz
        tracer[i, j, k] /= σⁿ(i, j, k, grid, Center(), Center(), Center())
    end
end
