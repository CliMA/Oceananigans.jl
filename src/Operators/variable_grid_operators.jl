import Oceananigans.Grids: znode, AbstractZStarGrid

#####
##### ZStar-specific vertical spacing functions
#####

const C = Center
const F = Face

const ZSG = AbstractZStarGrid

# Fallbacks
@inline e₃ⁿ(i, j, k, grid, ℓx, ℓy, ℓz) = one(grid)
@inline e₃⁻(i, j, k, grid, ℓx, ℓy, ℓz) = one(grid)

@inline e₃ⁿ(i, j, k, grid::ZSG, ::C, ::C, ℓz) = @inbounds grid.z.e₃ᶜᶜⁿ[i, j, 1]
@inline e₃ⁿ(i, j, k, grid::ZSG, ::F, ::C, ℓz) = @inbounds grid.z.e₃ᶠᶜⁿ[i, j, 1]
@inline e₃ⁿ(i, j, k, grid::ZSG, ::C, ::F, ℓz) = @inbounds grid.z.e₃ᶜᶠⁿ[i, j, 1]
@inline e₃ⁿ(i, j, k, grid::ZSG, ::F, ::F, ℓz) = @inbounds grid.z.e₃ᶠᶠⁿ[i, j, 1]

# e₃⁻ is needed only at centers
@inline e₃⁻(i, j, k, grid::ZSG, ::C, ::C, ℓz) = @inbounds grid.z.e₃ᶜᶜ⁻[i, j, 1]

@inline ∂t_e₃(i, j, k, grid)      = zero(grid)
@inline ∂t_e₃(i, j, k, grid::ZSG) = @inbounds grid.z.∂t_e₃[i, j, 1] 

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
        @inline $zspacing(i, j, k, grid::ZSRG)  = $rspacing(i, j, k, grid) * e₃ⁿ(i, j, k, grid, $ℓx(), $ℓy(), $ℓz())
        @inline $zspacing(i, j, k, grid::ZSLLG) = $rspacing(i, j, k, grid) * e₃ⁿ(i, j, k, grid, $ℓx(), $ℓy(), $ℓz())
        @inline $zspacing(i, j, k, grid::ZSOSG) = $rspacing(i, j, k, grid) * e₃ⁿ(i, j, k, grid, $ℓx(), $ℓy(), $ℓz())
    end
end

# rnode for an AbstractZStarGrid grid is scaled 
@inline znode(i, j, k, grid::ZSG, ::C, ::C, ℓz) = @inbounds rnode(i, j, k, grid, ℓx, ℓy, ℓz) * e₃ⁿ(i, j, k, grid, ℓx, ℓy, ℓz) + grid.z.ηⁿ[i, j, 1]
@inline znode(i, j, k, grid::ZSG, ::F, ::C, ℓz) = @inbounds rnode(i, j, k, grid, ℓx, ℓy, ℓz) * e₃ⁿ(i, j, k, grid, ℓx, ℓy, ℓz) +     ℑxᶠᵃᵃ(i, j, 1, grid, grid.z.ηⁿ)
@inline znode(i, j, k, grid::ZSG, ::C, ::F, ℓz) = @inbounds rnode(i, j, k, grid, ℓx, ℓy, ℓz) * e₃ⁿ(i, j, k, grid, ℓx, ℓy, ℓz) +     ℑyᵃᶠᵃ(i, j, 1, grid, grid.z.ηⁿ)
@inline znode(i, j, k, grid::ZSG, ::F, ::F, ℓz) = @inbounds rnode(i, j, k, grid, ℓx, ℓy, ℓz) * e₃ⁿ(i, j, k, grid, ℓx, ℓy, ℓz) +    ℑxyᶠᶠᵃ(i, j, 1, grid, grid.z.ηⁿ)
