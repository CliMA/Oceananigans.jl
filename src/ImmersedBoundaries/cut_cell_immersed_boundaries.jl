using Oceananigans.Fields: fill_halo_regions!
using Oceananigans.Architectures: arch_array
using Printf

#####
##### CutCellBottom
#####

struct CutCellBottom{H, E} <: AbstractGridFittedBottom{H}
    bottom_height :: H
    minimum_fractional_Δz :: E
end

function Base.summary(ib::CutCellBottom)
    hmax = maximum(ib.bottom_height)
    hmin = minimum(ib.bottom_height)
    return @sprintf("CutCellBottom(min(h)=%.2e, max(h)=%.2e, ϵ=%.1f)",
                    hmin, hmax, ib.minimum_fractional_Δz)
end

Base.summary(ib::CutCellBottom{<:Function}) = @sprintf("GridFittedBottom(%s, ϵ=%.1f)", ib.bottom_height, ib.minimum_fractional_Δz)


# TODO: nicer show method?
Base.show(io::IO, ib::CutCellBottom) = print(io, summary(ib))

"""
    CutCellBottom(bottom, minimum_height)

Return an immersed boundary...
"""
CutCellBottom(bottom_height; minimum_fractional_Δz=0.1) =
    CutCellBottom(bottom_height, minimum_fractional_Δz)

"""

        --x--
          ∘   k+1
    k+1 --x--    ↑      <- node z
          ∘   k  | Δz
      k --x--    ↓
      
Criterion is h >= z - ϵ Δz

"""
@inline function _immersed_cell(i, j, k, underlying_grid, ib::CutCellBottom)
    # Face node above current cell
    z = znode(i, j, k+1, underlying_grid, c, c, f)
    h = @inbounds ib.bottom_height[i, j]
    return z <= h
end

const SCIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:CutCellBottom}

on_architecture(arch, ib::CutCellBottom) = CutCellBottom(arch_array(arch, ib.bottom_height), ib.minimum_fractional_Δz)
Adapt.adapt_structure(to, ib::CutCellBottom) = CutCellBottom(adapt(to, ib.bottom_height), ib.minimum_fractional_Δz)     

bottom_cell(i, j, k, ibg::SCIBG) = !immersed_cell(i, j, k,   ibg.underlying_grid, ibg.immersed_boundary) &
                                    immersed_cell(i, j, k-1, ibg.underlying_grid, ibg.immersed_boundary)

@inline function Δzᶜᶠᶜ(i, j, k, ibg::SCIBG)
    underlying_grid = ibg.underlying_grid
    ib = ibg.immersed_boundary
    # Get node at face above and defining nodes on c,c,f
    x, y, z = node(i, j, k+1, underlying_grid, c, f, f)

    # Get bottom height and fractional Δz parameter
    h = ℑyᵃᶠᵃ(i, j, 1, ibg, ib.bottom_height)
    ϵ = ibg.immersed_boundary.minimum_fractional_Δz

    # Are we in a bottom cell?
    at_the_bottom = bottom_cell(i, j, k, ibg)

    full_Δz = Δzᶜᶠᶜ(i, j, k, ibg.underlying_grid)
    Cut_Δz = max(ϵ * full_Δz, z - h)

    return ifelse(at_the_bottom, Cut_Δz, full_Δz)
end

@inline function Δzᶜᶠᶠ(i, j, k, ibg::SCIBG)
    just_above_bottom = bottom_cell(i, j, k-1, ibg)
    zc = znode(i, j, k, ibg.underlying_grid, c, f, c)
    zf = znode(i, j, k, ibg.underlying_grid, c, f, f)

    full_Δz = Δzᶜᶠᶠ(i, j, k, ibg.underlying_grid)
    Cut_Δz = zc - zf + Δzᶜᶠᶜ(i, j, k-1, ibg) / 2

    Δz = ifelse(just_above_bottom, Cut_Δz, full_Δz)

    return Δz
end

@inline function Δzᶠᶜᶜ(i, j, k, ibg::SCIBG)
    underlying_grid = ibg.underlying_grid
    ib = ibg.immersed_boundary
    # Get node at face above and defining nodes on c,c,f
    x, y, z = node(i, j, k+1, underlying_grid, f, c, f)

    # Get bottom height and fractional Δz parameter
    h = ℑxᶠᵃᵃ(i, j, 1, ibg, ib.bottom_height)
    ϵ = ibg.immersed_boundary.minimum_fractional_Δz

    # Are we in a bottom cell?
    at_the_bottom = bottom_cell(i, j, k, ibg)

    full_Δz = Δzᶠᶜᶜ(i, j, k, ibg.underlying_grid)
    Cut_Δz = max(ϵ * full_Δz, z - h)

    return ifelse(at_the_bottom, Cut_Δz, full_Δz)
end

@inline function Δzᶠᶜᶠ(i, j, k, ibg::SCIBG)
    just_above_bottom = bottom_cell(i, j, k-1, ibg)
    zc = znode(i, j, k, ibg.underlying_grid, f, c, c)
    zf = znode(i, j, k, ibg.underlying_grid, f, c, f)

    full_Δz = Δzᶠᶜᶠ(i, j, k, ibg.underlying_grid)
    Cut_Δz = zc - zf + Δzᶠᶜᶜ(i, j, k-1, ibg) / 2

    Δz = ifelse(just_above_bottom, Cut_Δz, full_Δz)

    return Δz
end


@inline Δzᶜᶜᶜ(i, j, k, ibg::SCIBG) = (ℑxᶜᵃᵃ(i, j, k, ibg, Δzᶠᶜᶜ) + ℑyᵃᶜᵃ(i, j, k, ibg, Δzᶜᶠᶜ)) / 2
@inline Δzᶜᶜᶠ(i, j, k, ibg::SCIBG) = (ℑxᶜᵃᵃ(i, j, k, ibg, Δzᶠᶜᶠ) + ℑyᵃᶜᵃ(i, j, k, ibg, Δzᶜᶠᶠ)) / 2

@inline Δzᶠᶠᶜ(i, j, k, ibg::SCIBG) = (ℑxᶠᵃᵃ(i, j, k, ibg, Δzᶜᶠᶜ) + ℑyᵃᶠᵃ(i, j, k, ibg, Δzᶠᶜᶜ)) / 2
@inline Δzᶠᶠᶠ(i, j, k, ibg::SCIBG) = (ℑxᶠᵃᵃ(i, j, k, ibg, Δzᶜᶠᶠ) + ℑyᵃᶠᵃ(i, j, k, ibg, Δzᶠᶜᶠ)) / 2

@inline z_bottom(i, j, ibg::SCIBG) = @inbounds ibg.immersed_boundary.bottom_height[i, j]
