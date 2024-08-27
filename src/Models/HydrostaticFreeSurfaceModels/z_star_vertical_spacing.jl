
#####
##### ZStar coordinate and associated grid types
#####

""" free-surface following vertical coordinate """
struct ZStar end

"""
    struct ZStarSpacing{R, S} <: AbstractVerticalSpacing{R}

A vertical spacing for the hydrostatic free surface model that follows the free surface.
The vertical spacing is defined by a reference spacing `Δr` and a scaling `s` that obeys
```math
s = (η + H) / H
```
where ``η`` is the free surface height and ``H`` the vertical depth of the water column

# Fields
- `Δr`: reference vertical spacing with `η = 0`
- `sⁿ`: scaling of the vertical coordinate at time step `n`
- `s⁻`:scaling of the vertical coordinate at time step `n - 1`
- `∂t_∂s`: Time derivative of `s`
"""
struct ZStarSpacing{R, S} <: AbstractVerticalSpacing{R}
    Δr :: R
    sⁿ :: S
    s⁻ :: S
 ∂t_∂s :: S
end

Adapt.adapt_structure(to, coord::ZStarSpacing) = 
            ZStarSpacing(Adapt.adapt(to, coord.Δr),
                         Adapt.adapt(to, coord.s⁻),
                         Adapt.adapt(to, coord.sⁿ),
                         Adapt.adapt(to, coord.∂t_∂s))

on_architecture(arch, coord::ZStarSpacing) = 
            ZStarSpacing(on_architecture(arch, coord.Δr), 
                         on_architecture(arch, coord.s⁻),
                         on_architecture(arch, coord.sⁿ),
                         on_architecture(arch, coord.∂t_∂s))

Grids.coordinate_summary(Δ::ZStarSpacing, name) = 
    @sprintf("Free-surface following with Δ%s=%s", name, prettysummary(Δ.Δr))

const ZStarSpacingRG  = RectilinearGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:ZStarSpacing}
const ZStarSpacingLLG = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:ZStarSpacing} 
const ZStarSpacingOSG = OrthogonalSphericalShellGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:ZStarSpacing} 

const ZStarSpacingUnderlyingGrid = Union{ZStarSpacingRG, ZStarSpacingLLG, ZStarSpacingOSG}
const ZStarSpacingImmersedGrid   = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:ZStarSpacingUnderlyingGrid} 

const ZStarSpacingGrid = Union{ZStarSpacingUnderlyingGrid, ZStarSpacingImmersedGrid}

function generalized_spacing_grid(grid::ImmersedBoundaryGrid, ::ZStar)
    underlying_grid  = generalized_spacing_grid(grid.underlying_grid, ZStar())
    active_cells_map = !isnothing(grid.interior_active_cells)

    return ImmersedBoundaryGrid(underlying_grid, grid.immersed_boundary; active_cells_map)
end

# Replacing the z-coordinate with a moving vertical coordinate, defined by 
# - the reference spacing,
# - the scaling to apply to the reference
# - the derivative in time of the spacing
function generalized_spacing_grid(grid::AbstractUnderlyingGrid{FT, TX, TY, TZ}, ::ZStar) where {FT, TX, TY, TZ}
    
    s⁻    = Field{Center, Center, Nothing}(grid)
    sⁿ    = Field{Center, Center, Nothing}(grid)
    ∂t_∂s = Field{Center, Center, Nothing}(grid)
    
    # Initial "at-rest" conditions
    fill!(s⁻, 1)
    fill!(sⁿ, 1)

    Δzᵃᵃᶠ = ZStarSpacing(grid.Δzᵃᵃᶠ, s⁻, sⁿ, ∂t_∂s)
    Δzᵃᵃᶜ = ZStarSpacing(grid.Δzᵃᵃᶜ, s⁻, sⁿ, ∂t_∂s)

    args = []
    for prop in propertynames(grid)
        if prop == :Δzᵃᵃᶠ
            push!(args, Δzᵃᵃᶠ)
        elseif prop == :Δzᵃᵃᶜ
            push!(args, Δzᵃᵃᶜ)
        else
            push!(args, getproperty(grid, prop))
        end
    end

    GridType = getnamewrapper(grid)

    return GridType{TX, TY, TZ}(args...)
end

#####
##### ZStar-specific vertical spacing functions
#####

reference_Δzᵃᵃᶠ(grid::ZStarSpacingGrid) = grid.Δzᵃᵃᶠ.Δr
reference_Δzᵃᵃᶜ(grid::ZStarSpacingGrid) = grid.Δzᵃᵃᶜ.Δr

@inline Δzᶜᶜᶠ(i, j, k, grid::ZStarSpacingGrid) = @inbounds Δzᶜᶜᶠ_reference(i, j, k, grid) * grid.Δzᵃᵃᶠ.sⁿ[i, j, 1]
@inline Δzᶜᶜᶜ(i, j, k, grid::ZStarSpacingGrid) = @inbounds Δzᶜᶜᶜ_reference(i, j, k, grid) * grid.Δzᵃᵃᶜ.sⁿ[i, j, 1]

@inline ∂t_∂s_grid(i, j, k, grid::ZStarSpacingGrid) = grid.Δzᵃᵃᶜ.∂t_∂s[i, j, k] 
@inline V_times_∂t_∂s_grid(i, j, k, grid::ZStarSpacingGrid) = ∂t_∂s_grid(i, j, k, grid) * Vᶜᶜᶜ(i, j, k, grid)

#####
##### ZStar-specific vertical spacings update
#####

function update_vertical_spacing!(model, grid::ZStarSpacingGrid; parameters = :xy)
    
    # Scaling 
    s⁻    = grid.Δzᵃᵃᶠ.s⁻
    sⁿ    = grid.Δzᵃᵃᶠ.sⁿ
    ∂t_∂s = grid.Δzᵃᵃᶠ.∂t_∂s

    # Free surface variables
    Hᶜᶜ = model.free_surface.auxiliary.Hᶜᶜ
    U̅   = model.free_surface.state.U̅
    V̅   = model.free_surface.state.V̅
    η   = model.free_surface.η

    # Update vertical spacing with available parameters 
    # No need to fill the halo as the scaling is updated _IN_ the halos
    launch!(architecture(grid), grid, parameters, _update_zstar!, 
            sⁿ, s⁻, η, Hᶜᶜ, grid)
    
    # Update the time derivative of the grid-scaling. Note that in this case we leverage the
    # free surface evolution equation, where the time derivative of the free surface is equal
    # to the divergence of the vertically integrated velocity field, such that
    # ∂ₜ((H + η) / H) = H⁻¹ ∂ₜη =  - H⁻¹ ∇ ⋅ ∫udz 
    launch!(architecture(grid), grid, parameters, _update_∂t_∂s!, 
            ∂t_∂s, U̅, V̅, Hᶜᶜ, grid)

    return nothing
end

# NOTE: The ZStar vertical spacing works only for a SplitExplicitFreeSurface
@kernel function _update_∂t_∂s!(∂t_∂s, U̅, V̅, Hᶜᶜ, grid)
    i, j  = @index(Global, NTuple)
    k_top = grid.Nz + 1 
    @inbounds begin
        # ∂(η / H)/∂t = - ∇ ⋅ ∫udz / H
        ∂t_∂s[i, j, 1] = - div_xyᶜᶜᶜ(i, j, k_top - 1, grid, U̅, V̅) /  Hᶜᶜ[i, j, 1] 
    end
end

@kernel function _update_zstar!(sⁿ, s⁻, η, Hᶜᶜ, grid) 
    i, j = @index(Global, NTuple)
    @inbounds begin
        bottom = Hᶜᶜ[i, j, 1]
        h = (bottom + η[i, j, grid.Nz+1]) / bottom

        # update current and previous scaling
        s⁻[i, j, 1] = sⁿ[i, j, 1]
        sⁿ[i, j, 1] = h
    end
end

#####
##### ZStar-specific implementation of the additional terms to be included in the momentum equations
#####

@inline η_surfaceᶜᶜᶜ(i, j, k, grid, η, Hᶜᶜ) = @inbounds η[i, j, grid.Nz+1] * (1 + znode(i, j, k, grid, Center(), Center(), Center()) / Hᶜᶜ[i, j, 1])

@inline slope_xᶠᶜᶜ(i, j, k, grid, free_surface) = @inbounds ∂xᶠᶜᶜ(i, j, k, grid, η_surfaceᶜᶜᶜ, free_surface.η, free_surface.auxiliary.Hᶜᶜ)
@inline slope_yᶜᶠᶜ(i, j, k, grid, free_surface) = @inbounds ∂yᶜᶠᶜ(i, j, k, grid, η_surfaceᶜᶜᶜ, free_surface.η, free_surface.auxiliary.Hᶜᶜ)

@inline grid_slope_contribution_x(i, j, k, grid::ZStarSpacingGrid, free_surface, ::Nothing, model_fields) = zero(grid)
@inline grid_slope_contribution_y(i, j, k, grid::ZStarSpacingGrid, free_surface, ::Nothing, model_fields) = zero(grid)

@inline grid_slope_contribution_x(i, j, k, grid::ZStarSpacingGrid, free_surface, buoyancy, model_fields) = 
    ℑxᶠᵃᵃ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, buoyancy.model, model_fields) * slope_xᶠᶜᶜ(i, j, k, grid, free_surface)

@inline grid_slope_contribution_y(i, j, k, grid::ZStarSpacingGrid, free_surface, buoyancy, model_fields) = 
    ℑyᵃᶠᵃ(i, j, k, grid, buoyancy_perturbationᶜᶜᶜ, buoyancy.model, model_fields) * slope_yᶜᶠᶜ(i, j, k, grid, free_surface)
