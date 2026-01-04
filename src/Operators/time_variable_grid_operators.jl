import Oceananigans.Grids: znode, AbstractGeneralizedVerticalGrid

#####
##### ZStarVerticalCoordinate-specific vertical spacing functions
#####
##### These operators implement the coordinate transformation from computational
##### coordinates (ξ, η, r) to physical coordinates (x, y, z).
#####
##### The key relationships are:
#####   z(ξ, η, r, t) = η_fs(ξ, η, t) + σ(ξ, η, t) * r
#####   σ = ∂z/∂r = (H + η_fs) / H
#####
##### where η_fs is the free surface displacement (stored as ηⁿ in code).
#####

# Shorthand alias for generalized vertical grids
const AGG = AbstractGeneralizedVerticalGrid

# Fallbacks for static grids (identity mapping: z = r, σ = 1)
@inline σⁿ(i, j, k, grid, ℓx, ℓy, ℓz) = one(grid)
@inline σ⁻(i, j, k, grid, ℓx, ℓy, ℓz) = one(grid)

# Specific thickness σ = ∂z/∂r at various staggered locations for generalized grids
@inline σⁿ(i, j, k, grid::AGG, ::C, ::C, ℓz) = @inbounds grid.z.σᶜᶜⁿ[i, j, 1]
@inline σⁿ(i, j, k, grid::AGG, ::F, ::C, ℓz) = @inbounds grid.z.σᶠᶜⁿ[i, j, 1]
@inline σⁿ(i, j, k, grid::AGG, ::C, ::F, ℓz) = @inbounds grid.z.σᶜᶠⁿ[i, j, 1]
@inline σⁿ(i, j, k, grid::AGG, ::F, ::F, ℓz) = @inbounds grid.z.σᶠᶠⁿ[i, j, 1]

# σ⁻ is needed only at centers (for time-stepping)
@inline σ⁻(i, j, k, grid::AGG, ::C, ::C, ℓz) = @inbounds grid.z.σᶜᶜ⁻[i, j, 1]

# Time derivative of specific thickness
@inline ∂t_σ(i, j, k, grid)      = zero(grid)
@inline ∂t_σ(i, j, k, grid::AGG) = @inbounds grid.z.∂t_σ[i, j, 1]

####
#### Vertical spacing functions: Δz = σ * Δr
####

const GRG = RectilinearGrid{<:Any, <:Any, <:Any, <:Any, <:ZStarVerticalCoordinate}
const GLLG = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:ZStarVerticalCoordinate}
const GOSG = OrthogonalSphericalShellGrid{<:Any, <:Any, <:Any, <:Any, <:ZStarVerticalCoordinate}

superscript_location(s::Symbol) = s == :ᶜ ? :Center : :Face

for LX in (:ᶠ, :ᶜ), LY in (:ᶠ, :ᶜ), LZ in (:ᶠ, :ᶜ)
    zspacing = Symbol(:Δz, LX, LY, LZ)
    rspacing = Symbol(:Δr, LX, LY, LZ)

    ℓx = superscript_location(LX)
    ℓy = superscript_location(LY)
    ℓz = superscript_location(LZ)

    @eval begin
        @inline $zspacing(i, j, k, grid::GRG)  = $rspacing(i, j, k, grid) * σⁿ(i, j, k, grid, $ℓx(), $ℓy(), $ℓz())
        @inline $zspacing(i, j, k, grid::GLLG) = $rspacing(i, j, k, grid) * σⁿ(i, j, k, grid, $ℓx(), $ℓy(), $ℓz())
        @inline $zspacing(i, j, k, grid::GOSG) = $rspacing(i, j, k, grid) * σⁿ(i, j, k, grid, $ℓx(), $ℓy(), $ℓz())
    end
end

#####
##### Physical z-coordinate for generalized vertical grids
#####
##### The mapping from reference coordinate r to physical coordinate z is:
#####   z(ξ, η, r, t) = η_fs(ξ, η, t) + σ(ξ, η, t) * r
#####
##### where η_fs is the free surface displacement (stored as ηⁿ in code).
#####

@inline znode(i, j, k, grid::AGG, ::C, ::C, ℓz) = rnode(i, j, k, grid, C(), C(), ℓz) * σⁿ(i, j, k, grid, C(), C(), ℓz) + @inbounds grid.z.ηⁿ[i, j, 1]
@inline znode(i, j, k, grid::AGG, ::F, ::C, ℓz) = rnode(i, j, k, grid, F(), C(), ℓz) * σⁿ(i, j, k, grid, F(), C(), ℓz) +  ℑxᶠᵃᵃ(i, j, 1, grid, grid.z.ηⁿ)
@inline znode(i, j, k, grid::AGG, ::C, ::F, ℓz) = rnode(i, j, k, grid, C(), F(), ℓz) * σⁿ(i, j, k, grid, C(), F(), ℓz) +  ℑyᵃᶠᵃ(i, j, 1, grid, grid.z.ηⁿ)
@inline znode(i, j, k, grid::AGG, ::F, ::F, ℓz) = rnode(i, j, k, grid, F(), F(), ℓz) * σⁿ(i, j, k, grid, F(), F(), ℓz) + ℑxyᶠᶠᵃ(i, j, 1, grid, grid.z.ηⁿ)
