import Oceananigans.Grids: znode, AbstractMutableGrid

#####
##### MutableVerticalDiscretization-specific vertical spacing functions
#####

const AMG = AbstractMutableGrid

# Fallbacks
@inline σⁿ(i, j, k, grid, ℓx, ℓy, ℓz) = one(grid)
@inline σ⁻(i, j, k, grid, ℓx, ℓy, ℓz) = one(grid)

@inline σⁿ(i, j, k, grid::AMG, ::C, ::C, ℓz) = @inbounds grid.z.σᶜᶜⁿ[i, j, 1]
@inline σⁿ(i, j, k, grid::AMG, ::F, ::C, ℓz) = @inbounds grid.z.σᶠᶜⁿ[i, j, 1]
@inline σⁿ(i, j, k, grid::AMG, ::C, ::F, ℓz) = @inbounds grid.z.σᶜᶠⁿ[i, j, 1]
@inline σⁿ(i, j, k, grid::AMG, ::F, ::F, ℓz) = @inbounds grid.z.σᶠᶠⁿ[i, j, 1]

# σ⁻ is needed only at centers
@inline σ⁻(i, j, k, grid::AMG, ::C, ::C, ℓz) = @inbounds grid.z.σᶜᶜ⁻[i, j, 1]

@inline ∂t_σ(i, j, k, grid)      = zero(grid)
@inline ∂t_σ(i, j, k, grid::AMG) = @inbounds grid.z.∂t_σ[i, j, 1]

####
#### Vertical spacing functions
####

const MRG = RectilinearGrid{<:Any, <:Any, <:Any, <:Any, <:MutableVerticalDiscretization}
const MLLG = LatitudeLongitudeGrid{<:Any, <:Any, <:Any, <:Any, <:MutableVerticalDiscretization}
const MOSG = OrthogonalSphericalShellGrid{<:Any, <:Any, <:Any, <:Any, <:MutableVerticalDiscretization}

superscript_location(s::Symbol) = s == :ᶜ ? :Center : :Face

for LX in (:ᶠ, :ᶜ), LY in (:ᶠ, :ᶜ), LZ in (:ᶠ, :ᶜ)
    zspacing = Symbol(:Δz, LX, LY, LZ)
    rspacing = Symbol(:Δr, LX, LY, LZ)

    ℓx = superscript_location(LX)
    ℓy = superscript_location(LY)
    ℓz = superscript_location(LZ)

    @eval begin
        @inline $zspacing(i, j, k, grid::MRG)  = $rspacing(i, j, k, grid) * σⁿ(i, j, k, grid, $ℓx(), $ℓy(), $ℓz())
        @inline $zspacing(i, j, k, grid::MLLG) = $rspacing(i, j, k, grid) * σⁿ(i, j, k, grid, $ℓx(), $ℓy(), $ℓz())
        @inline $zspacing(i, j, k, grid::MOSG) = $rspacing(i, j, k, grid) * σⁿ(i, j, k, grid, $ℓx(), $ℓy(), $ℓz())
    end
end

# znode for an AbstractMutableGrid grid is the reference node (`rnode`) scaled by the derivative with respect to the reference (σⁿ)
# added to the surface value of `z` (which we here call ηⁿ)
@inline znode(i, j, k, grid::AMG, ::C, ::C, ℓz) = rnode(i, j, k, grid, C(), C(), ℓz) * σⁿ(i, j, k, grid, C(), C(), ℓz) + @inbounds grid.z.ηⁿ[i, j, 1]
@inline znode(i, j, k, grid::AMG, ::F, ::C, ℓz) = rnode(i, j, k, grid, F(), C(), ℓz) * σⁿ(i, j, k, grid, F(), C(), ℓz) +  ℑxᶠᵃᵃ(i, j, 1, grid, grid.z.ηⁿ)
@inline znode(i, j, k, grid::AMG, ::C, ::F, ℓz) = rnode(i, j, k, grid, C(), F(), ℓz) * σⁿ(i, j, k, grid, C(), F(), ℓz) +  ℑyᵃᶠᵃ(i, j, 1, grid, grid.z.ηⁿ)
@inline znode(i, j, k, grid::AMG, ::F, ::F, ℓz) = rnode(i, j, k, grid, F(), F(), ℓz) * σⁿ(i, j, k, grid, F(), F(), ℓz) + ℑxyᶠᶠᵃ(i, j, 1, grid, grid.z.ηⁿ)
