import Oceananigans.Grids: znode, AbstractZStarGrid

#####
##### ZStar-specific vertical spacing functions
#####

const C = Center
const F = Face

const ZSG = AbstractZStarGrid

# Fallbacks
@inline σⁿ(i, j, k, grid, ℓx, ℓy, ℓz) = one(grid)
@inline σ⁻(i, j, k, grid, ℓx, ℓy, ℓz) = one(grid)

@inline σⁿ(i, j, k, grid::ZSG, ::C, ::C, ℓz) = @inbounds grid.z.σᶜᶜⁿ[i, j, 1]
@inline σⁿ(i, j, k, grid::ZSG, ::F, ::C, ℓz) = @inbounds grid.z.σᶠᶜⁿ[i, j, 1]
@inline σⁿ(i, j, k, grid::ZSG, ::C, ::F, ℓz) = @inbounds grid.z.σᶜᶠⁿ[i, j, 1]
@inline σⁿ(i, j, k, grid::ZSG, ::F, ::F, ℓz) = @inbounds grid.z.σᶠᶠⁿ[i, j, 1]

# σ⁻ is needed only at centers
@inline σ⁻(i, j, k, grid::ZSG, ::C, ::C, ℓz) = @inbounds grid.z.σᶜᶜ⁻[i, j, 1]

@inline ∂t_σ(i, j, k, grid)      = zero(grid)
@inline ∂t_σ(i, j, k, grid::ZSG) = @inbounds grid.z.∂t_σ[i, j, 1] 

####
#### Vertical spacing functions
####

const ZSRG = RectilinearGrid{<:Any, <:Any, <:Any, <:Any, <:ZStarVerticalCoordinate}
const ZSLLG = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:ZStarVerticalCoordinate}
const ZSOSG = OrthogonalSphericalShellGrid{<:Any, <:Any, <:Any, <:Any, <:ZStarVerticalCoordinate}

superscript_location(s::Symbol) = s == :ᶜ ? :Center : :Face

for LX in (:ᶠ, :ᶜ), LY in (:ᶠ, :ᶜ), LZ in (:ᶠ, :ᶜ)
    zspacing = Symbol(:Δz, LX, LY, LZ)
    rspacing = Symbol(:Δr, LX, LY, LZ)

    ℓx = superscript_location(LX)
    ℓy = superscript_location(LY)
    ℓz = superscript_location(LZ)

    @eval begin
        @inline $zspacing(i, j, k, grid::ZSRG)  = $rspacing(i, j, k, grid) * σⁿ(i, j, k, grid, $ℓx(), $ℓy(), $ℓz())
        @inline $zspacing(i, j, k, grid::ZSLLG) = $rspacing(i, j, k, grid) * σⁿ(i, j, k, grid, $ℓx(), $ℓy(), $ℓz())
        @inline $zspacing(i, j, k, grid::ZSOSG) = $rspacing(i, j, k, grid) * σⁿ(i, j, k, grid, $ℓx(), $ℓy(), $ℓz())
    end
end

# znode for an AbstractZStarGrid grid is scaled by the free surface
@inline znode(i, j, k, grid::ZSG, ::C, ::C, ℓz) = @inbounds rnode(i, j, k, grid, C(), C(), ℓz) * σⁿ(i, j, k, grid, C(), C(), ℓz) + grid.z.ηⁿ[i, j, 1]
@inline znode(i, j, k, grid::ZSG, ::F, ::C, ℓz) = @inbounds rnode(i, j, k, grid, F(), C(), ℓz) * σⁿ(i, j, k, grid, F(), C(), ℓz) +     ℑxᶠᵃᵃ(i, j, 1, grid, grid.z.ηⁿ)
@inline znode(i, j, k, grid::ZSG, ::C, ::F, ℓz) = @inbounds rnode(i, j, k, grid, C(), F(), ℓz) * σⁿ(i, j, k, grid, C(), F(), ℓz) +     ℑyᵃᶠᵃ(i, j, 1, grid, grid.z.ηⁿ)
@inline znode(i, j, k, grid::ZSG, ::F, ::F, ℓz) = @inbounds rnode(i, j, k, grid, F(), F(), ℓz) * σⁿ(i, j, k, grid, F(), F(), ℓz) +    ℑxyᶠᶠᵃ(i, j, 1, grid, grid.z.ηⁿ)
