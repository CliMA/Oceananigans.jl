using Oceananigans.Grids

using Oceananigans.ImmersedBoundaries: ZStarGridOfSomeKind

using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces: dynamic_column_depthᶜᶜᵃ,
                                                                                  dynamic_column_depthᶜᶠᵃ,
                                                                                  dynamic_column_depthᶠᶜᵃ,
                                                                                  dynamic_column_depthᶠᶠᵃ

#####
##### ZStar-specific vertical spacings update
#####

function update_grid!(model, grid::ZStarGridOfSomeKind; parameters = :xy)

    # Scaling (just update once, they are the same for all the metrics)
    e₃ᶜᶜ⁻  = grid.z.e₃ᶜᶜ⁻
    e₃ᶜᶜⁿ  = grid.z.e₃ᶜᶜⁿ
    e₃ᶠᶜⁿ  = grid.z.e₃ᶠᶜⁿ
    e₃ᶜᶠⁿ  = grid.z.e₃ᶜᶠⁿ
    e₃ᶠᶠⁿ  = grid.z.e₃ᶠᶠⁿ
    ∂t_e₃  = grid.z.∂t_e₃
    ηⁿ     = grid.z.ηⁿ

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
            e₃ᶜᶜⁿ, e₃ᶠᶜⁿ, e₃ᶜᶠⁿ, e₃ᶠᶠⁿ, e₃ᶜᶜ⁻, ηⁿ, ∂t_e₃, grid, η, U, V)

    return nothing
end

# NOTE: The ZStar vertical spacing only supports a SplitExplicitFreeSurface
# TODO: extend to support other free surface solvers
@kernel function _update_zstar!(e₃ᶜᶜⁿ, e₃ᶠᶜⁿ, e₃ᶜᶠⁿ, e₃ᶠᶠⁿ, e₃ᶜᶜ⁻, ηⁿ, ∂t_e₃, grid, η, U, V)
    i, j = @index(Global, NTuple)
    kᴺ = size(grid, 3)

    hᶜᶜ = static_column_depthᶜᶜᵃ(i, j, grid)
    hᶠᶜ = static_column_depthᶠᶜᵃ(i, j, grid)
    hᶜᶠ = static_column_depthᶜᶠᵃ(i, j, grid)
    hᶠᶠ = static_column_depthᶠᶠᵃ(i, j, grid)

    Hᶜᶜ = dynamic_column_depthᶜᶜᵃ(i, j, grid, η)
    Hᶠᶜ = dynamic_column_depthᶠᶜᵃ(i, j, grid, η)
    Hᶜᶠ = dynamic_column_depthᶜᶠᵃ(i, j, grid, η)
    Hᶠᶠ = dynamic_column_depthᶠᶠᵃ(i, j, grid, η)

    @inbounds begin
        e₃ᶜᶜ = ifelse(hᶜᶜ == 0, one(grid), Hᶜᶜ / hᶜᶜ)
        e₃ᶠᶜ = ifelse(hᶠᶜ == 0, one(grid), Hᶠᶜ / hᶠᶜ)
        e₃ᶜᶠ = ifelse(hᶜᶠ == 0, one(grid), Hᶜᶠ / hᶜᶠ)
        e₃ᶠᶠ = ifelse(hᶠᶠ == 0, one(grid), Hᶠᶠ / hᶠᶠ)

        # Update previous scaling
        e₃ᶜᶜ⁻[i, j, 1] = e₃ᶜᶜⁿ[i, j, 1]
        
        # update current scaling
        e₃ᶜᶜⁿ[i, j, 1] = e₃ᶜᶜ
        e₃ᶠᶜⁿ[i, j, 1] = e₃ᶠᶜ
        e₃ᶜᶠⁿ[i, j, 1] = e₃ᶜᶠ
        e₃ᶠᶠⁿ[i, j, 1] = e₃ᶠᶠ

        # Update η in the grid
        ηⁿ[i, j, 1] = η[i, j, kᴺ+1]

        # ∂(η / H)/∂t = - ∇ ⋅ ∫udz / H
        δx_U = δxᶜᶜᶜ(i, j, kᴺ, grid, Δy_qᶠᶜᶜ, U)
        δy_V = δyᶜᶜᶜ(i, j, kᴺ, grid, Δx_qᶜᶠᶜ, V)

        δh_U = (δx_U + δy_V) / Azᶜᶜᶜ(i, j, kᴺ, grid)

        ∂t_e₃[i, j, 1] = ifelse(hᶜᶜ == 0, zero(grid), - δh_U / hᶜᶜ)
    end
end

#####
##### ZStar-specific implementation of the additional terms to be included in the momentum equations
#####

@inline ∂x_z(i, j, k, grid) = @inbounds ∂xᶠᶜᶜ(i, j, k, grid, znode)
@inline ∂y_z(i, j, k, grid) = @inbounds ∂yᶜᶠᶜ(i, j, k, grid, znode)

@inline grid_slope_contribution_x(i, j, k, grid::ZStarGridOfSomeKind, ::Nothing, model_fields) = zero(grid)
@inline grid_slope_contribution_y(i, j, k, grid::ZStarGridOfSomeKind, ::Nothing, model_fields) = zero(grid)

@inline grid_slope_contribution_x(i, j, k, grid::ZStarGridOfSomeKind, buoyancy, model_fields) = 
    ℑxᶠᵃᵃ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, buoyancy.model, model_fields) * ∂x_z(i, j, k, grid)

@inline grid_slope_contribution_y(i, j, k, grid::ZStarGridOfSomeKind, buoyancy, model_fields) = 
    ℑyᵃᶠᵃ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, buoyancy.model, model_fields) * ∂y_z(i, j, k, grid)

####
#### Removing the scaling of the vertical coordinate from the tracer fields
####

const EmptyTuples = Union{NamedTuple{(), Tuple{}}, Tuple{}}

unscale_tracers!(::EmptyTuples, ::ZStarGridOfSomeKind; kwargs...) = nothing

tracer_scaling_parameters(param::Symbol, tracers, grid) = KernelParameters((size(grid, 1), size(grid, 2), length(tracers)), (0, 0, 0))
tracer_scaling_parameters(param::KernelParameters{S, O}, tracers, grid) where {S, O} = KernelParameters((S..., length(tracers)), (O..., 0))

function unscale_tracers!(tracers, grid::ZStarGridOfSomeKind; parameters = :xy) 
    parameters = tracer_scaling_parameters(parameters, tracers, grid)
    
    launch!(architecture(grid), grid, parameters, _unscale_tracers!, tracers, grid, 
            Val(grid.Hz), Val(grid.Nz))
    
    return nothing
end
    
@kernel function _unscale_tracers!(tracers, grid, ::Val{Hz}, ::Val{Nz}) where {Hz, Nz}
    i, j, n = @index(Global, NTuple)

    @unroll for k in -Hz+1:Nz+Hz
        tracers[n][i, j, k] /= e₃ⁿ(i, j, k, grid, Center(), Center(), Center())
    end
end