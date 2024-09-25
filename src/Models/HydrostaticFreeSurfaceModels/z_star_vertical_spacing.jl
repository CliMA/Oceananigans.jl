
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
- `sᶜᶜⁿ`: scaling of the vertical coordinate at time step `n` at `(Center, Center, Any)` location
- `sᶠᶜⁿ`: scaling of the vertical coordinate at time step `n` at `(Face,   Center, Any)` location
- `sᶜᶠⁿ`: scaling of the vertical coordinate at time step `n` at `(Center, Face,   Any)` location
- `sᶠᶠⁿ`: scaling of the vertical coordinate at time step `n` at `(Face,   Face,   Any)` location
- `s⁻`: scaling of the vertical coordinate at time step `n - 1` at `(Center, Center, Any)` location
- `∂t_∂s`: Time derivative of `s`
"""
struct ZStarSpacing{R, SCC, SFC, SCF, SFF} <: AbstractVerticalSpacing{R}
      Δr :: R
    sᶜᶜⁿ :: SCC
    sᶠᶜⁿ :: SFC
    sᶜᶠⁿ :: SCF
    sᶠᶠⁿ :: SFF
    sᶜᶜ⁻ :: SCC
    sᶠᶜ⁻ :: SFC
    sᶜᶠ⁻ :: SCF
   ∂t_∂s :: SCC
end

Adapt.adapt_structure(to, coord::ZStarSpacing) = 
            ZStarSpacing(Adapt.adapt(to, coord.Δr),
                         Adapt.adapt(to, coord.sᶜᶜⁿ),
                         Adapt.adapt(to, coord.sᶠᶜⁿ),
                         Adapt.adapt(to, coord.sᶜᶠⁿ),
                         Adapt.adapt(to, coord.sᶠᶠⁿ),
                         Adapt.adapt(to, coord.sᶜᶜ⁻),
                         Adapt.adapt(to, coord.sᶠᶜ⁻),
                         Adapt.adapt(to, coord.sᶜᶠ⁻),
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
    
    sᶜᶜ⁻  = Field{Center, Center, Nothing}(grid)
    sᶜᶜⁿ  = Field{Center, Center, Nothing}(grid)
    sᶠᶜ⁻  = Field{Face,   Center, Nothing}(grid)
    sᶠᶜⁿ  = Field{Face,   Center, Nothing}(grid)
    sᶜᶠ⁻  = Field{Center, Face,   Nothing}(grid)
    sᶜᶠⁿ  = Field{Center, Face,   Nothing}(grid)
    sᶠᶠⁿ  = Field{Face,   Face,   Nothing}(grid)
    ∂t_∂s = Field{Center, Center, Nothing}(grid)
    
    # Initial "at-rest" conditions
    fill!(sᶜᶜ⁻, 1)
    fill!(sᶜᶜⁿ, 1)
    fill!(sᶠᶜⁿ, 1)
    fill!(sᶜᶠⁿ, 1)
    fill!(sᶠᶜ⁻, 1)
    fill!(sᶜᶠ⁻, 1)
    fill!(sᶠᶠⁿ, 1)

    Δzᵃᵃᶠ = ZStarSpacing(grid.Δzᵃᵃᶠ, sᶜᶜⁿ, sᶠᶜⁿ, sᶜᶠⁿ, sᶠᶠⁿ, sᶜᶜ⁻, sᶠᶜ⁻, sᶜᶠ⁻, ∂t_∂s)
    Δzᵃᵃᶜ = ZStarSpacing(grid.Δzᵃᵃᶜ, sᶜᶜⁿ, sᶠᶜⁿ, sᶜᶠⁿ, sᶠᶠⁿ, sᶜᶜ⁻, sᶠᶜ⁻, sᶜᶠ⁻, ∂t_∂s)

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

@inline vertical_scaling(i, j, k, grid, ℓx, ℓy, ℓz) = one(grid)
@inline vertical_scaling(i, j, k, grid, ℓx, ℓy, ℓz) = one(grid)
@inline vertical_scaling(i, j, k, grid, ℓx, ℓy, ℓz) = one(grid)

@inline vertical_scaling(i, j, k, grid::ZStarSpacingGrid, ::Center, ::Center, ℓz) = @inbounds grid.Δzᵃᵃᶠ.sᶜᶜⁿ[i, j, 1]
@inline vertical_scaling(i, j, k, grid::ZStarSpacingGrid, ::Face,   ::Center, ℓz) = @inbounds grid.Δzᵃᵃᶠ.sᶠᶜⁿ[i, j, 1]
@inline vertical_scaling(i, j, k, grid::ZStarSpacingGrid, ::Center, ::Face, ℓz)   = @inbounds grid.Δzᵃᵃᶠ.sᶜᶠⁿ[i, j, 1]

@inline previous_vertical_scaling(i, j, k, grid, ℓx, ℓy, ℓz) = one(grid)
@inline previous_vertical_scaling(i, j, k, grid, ℓx, ℓy, ℓz) = one(grid)
@inline previous_vertical_scaling(i, j, k, grid, ℓx, ℓy, ℓz) = one(grid)

@inline previous_vertical_scaling(i, j, k, grid::ZStarSpacingGrid, ::Center, ::Center, ℓz) = @inbounds grid.Δzᵃᵃᶠ.sᶜᶜ⁻[i, j, 1]
@inline previous_vertical_scaling(i, j, k, grid::ZStarSpacingGrid, ::Face,   ::Center, ℓz) = @inbounds grid.Δzᵃᵃᶠ.sᶠᶜ⁻[i, j, 1]
@inline previous_vertical_scaling(i, j, k, grid::ZStarSpacingGrid, ::Center, ::Face, ℓz)   = @inbounds grid.Δzᵃᵃᶠ.sᶜᶠ⁻[i, j, 1]

reference_zspacings(grid::ZStarSpacingGrid, ::Face)   = grid.Δzᵃᵃᶠ.Δr
reference_zspacings(grid::ZStarSpacingGrid, ::Center) = grid.Δzᵃᵃᶜ.Δr

@inline ∂t_∂s_grid(i, j, k, grid::ZStarSpacingGrid) = grid.Δzᵃᵃᶜ.∂t_∂s[i, j, 1] 
@inline V_times_∂t_∂s_grid(i, j, k, grid::ZStarSpacingGrid) = ∂t_∂s_grid(i, j, k, grid) * Vᶜᶜᶜ(i, j, k, grid)

@inline rnode(i, j, k, grid, ℓx, ℓy, ::Face)   = getnode(grid.zᵃᵃᶠ, k)
@inline rnode(i, j, k, grid, ℓx, ℓy, ::Center) = getnode(grid.zᵃᵃᶜ, k)

#####
##### ZStar-specific vertical spacings update
#####

function update_vertical_spacing!(model, grid::ZStarSpacingGrid; parameters = :xy)
    
    # Scaling 
    sᶜᶜ⁻  = grid.Δzᵃᵃᶠ.sᶜᶜ⁻
    sᶜᶜⁿ  = grid.Δzᵃᵃᶠ.sᶜᶜⁿ
    sᶠᶜ⁻  = grid.Δzᵃᵃᶠ.sᶠᶜ⁻
    sᶠᶜⁿ  = grid.Δzᵃᵃᶠ.sᶠᶜⁿ
    sᶜᶠ⁻  = grid.Δzᵃᵃᶠ.sᶜᶠ⁻
    sᶜᶠⁿ  = grid.Δzᵃᵃᶠ.sᶜᶠⁿ
    sᶠᶠⁿ  = grid.Δzᵃᵃᶠ.sᶠᶠⁿ
    ∂t_∂s = grid.Δzᵃᵃᶠ.∂t_∂s

    # Free surface variables
    Hᶜᶜ = model.free_surface.auxiliary.Hᶜᶜ
    Hᶠᶜ = model.free_surface.auxiliary.Hᶠᶜ
    Hᶜᶠ = model.free_surface.auxiliary.Hᶜᶠ
    Hᶠᶠ = model.free_surface.auxiliary.Hᶠᶠ
    U̅   = model.free_surface.state.U̅ # Ũ
    V̅   = model.free_surface.state.V̅ # Ṽ
    η   = model.free_surface.η

    # Update vertical spacing with available parameters 
    # No need to fill the halo as the scaling is updated _IN_ the halos
    launch!(architecture(grid), grid, parameters, _update_zstar!, 
            sᶜᶜⁿ, sᶠᶜⁿ, sᶜᶠⁿ, sᶠᶠⁿ, sᶜᶜ⁻, sᶠᶜ⁻, sᶜᶠ⁻, η, Hᶜᶜ, Hᶠᶜ, Hᶜᶠ, Hᶠᶠ, grid)
    
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

@kernel function _update_zstar!(sᶜᶜⁿ, sᶠᶜⁿ, sᶜᶠⁿ, sᶠᶠⁿ, sᶜᶜ⁻, sᶠᶜ⁻, sᶜᶠ⁻, η, Hᶜᶜ, Hᶠᶜ, Hᶜᶠ, Hᶠᶠ, grid)
    i, j = @index(Global, NTuple)
    k_top = grid.Nz+1
    @inbounds begin
        hᶜᶜ = (Hᶜᶜ[i, j, 1] + η[i, j, k_top]) / Hᶜᶜ[i, j, 1]
        hᶠᶜ = (Hᶠᶜ[i, j, 1] +  ℑxᶠᵃᵃ(i, j, k_top, grid, η)) / Hᶠᶜ[i, j, 1]
        hᶜᶠ = (Hᶜᶠ[i, j, 1] +  ℑyᵃᶠᵃ(i, j, k_top, grid, η)) / Hᶜᶠ[i, j, 1]
        hᶠᶠ = (Hᶠᶠ[i, j, 1] + ℑxyᶠᶠᵃ(i, j, k_top, grid, η)) / Hᶠᶠ[i, j, 1]

        sᶜᶜ⁻[i, j, 1] = sᶜᶜⁿ[i, j, 1]
        sᶠᶜ⁻[i, j, 1] = sᶠᶜⁿ[i, j, 1]
        sᶜᶠ⁻[i, j, 1] = sᶜᶠⁿ[i, j, 1]
        
        # update current and previous scaling
        sᶜᶜⁿ[i, j, 1] = hᶜᶜ
        sᶠᶜⁿ[i, j, 1] = hᶠᶜ
        sᶜᶠⁿ[i, j, 1] = hᶜᶠ
        sᶠᶠⁿ[i, j, 1] = hᶠᶠ
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
