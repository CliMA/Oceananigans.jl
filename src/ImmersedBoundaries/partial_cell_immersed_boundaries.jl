using Oceananigans.Fields: fill_halo_regions!
using Oceananigans.Architectures: arch_array
using Printf

#####
##### PartialCellBottom
#####

struct PartialCellBottom{H, E} <: AbstractGridFittedBottom{H}
    bottom_height :: H
    minimum_fractional_Δz :: E
end

function Base.summary(ib::PartialCellBottom)
    hmax = maximum(ib.bottom_height)
    hmin = minimum(ib.bottom_height)
    return @sprintf("PartialCellBottom(min(h)=%.2e, max(h)=%.2e, ϵ=%.1f)",
                    hmin, hmax, ib.minimum_fractional_Δz)
end

Base.summary(ib::PartialCellBottom{<:Function}) = @sprintf("GridFittedBottom(%s, ϵ=%.1f)", ib.bottom_height, ib.minimum_fractional_Δz)


# TODO: nicer show method?
Base.show(io::IO, ib::PartialCellBottom) = print(io, summary(ib))

"""
    PartialCellBottom(bottom, minimum_height)

Return an immersed boundary...
"""
PartialCellBottom(bottom_height; minimum_fractional_Δz=0.1) =
    PartialCellBottom(bottom_height, minimum_fractional_Δz)

"""

        --x--
          ∘   k+1
    k+1 --x--    ↑      <- node z
          ∘   k  | Δz
      k --x--    ↓
      
Criterion is h >= z - ϵ Δz

"""
@inline function _immersed_cell(i, j, k, underlying_grid, ib::PartialCellBottom)
    # Face node above current cell
    z = znode(c, c, f, i, j, k+1, underlying_grid)
    h = @inbounds ib.bottom_height[i, j]
    return z <= h
end

const PCIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:PartialCellBottom}

on_architecture(arch, ib::PartialCellBottom) = PartialCellBottom(arch_array(arch, ib.bottom_height), ib.minimum_fractional_Δz)
Adapt.adapt_structure(to, ib::PartialCellBottom) = PartialCellBottom(adapt(to, ib.bottom_height), ib.minimum_fractional_Δz)     

bottom_cell(i, j, k, ibg::PCIBG) = !immersed_cell(i, j, k,   ibg.underlying_grid, ibg.immersed_boundary) &
                                    immersed_cell(i, j, k-1, ibg.underlying_grid, ibg.immersed_boundary)

@inline function Δzᶜᶜᶜ(i, j, k, ibg::PCIBG)
    underlying_grid = ibg.underlying_grid
    ib = ibg.immersed_boundary
    # Get node at face above and defining nodes on c,c,f
    x, y, z = node(c, c, f, i, j, k+1, underlying_grid)

    # Get bottom height and fractional Δz parameter
    h = @inbounds ib.bottom_height[i, j]
    ϵ = ibg.immersed_boundary.minimum_fractional_Δz

    # Are we in a bottom cell?
    at_the_bottom = bottom_cell(i, j, k, ibg)

    full_Δz = Δzᶜᶜᶜ(i, j, k, ibg.underlying_grid)
    partial_Δz = max(ϵ * full_Δz, z - h)

    return ifelse(at_the_bottom, partial_Δz, full_Δz)
end

@inline function Δzᶜᶜᶠ(i, j, k, ibg::PCIBG)
    just_above_bottom = bottom_cell(i, j, k-1, ibg)
    zc = znode(c, c, c, i, j, k, ibg.underlying_grid)
    zf = znode(c, c, f, i, j, k, ibg.underlying_grid)

    full_Δz = Δzᶜᶜᶠ(i, j, k, ibg.underlying_grid)
    partial_Δz = zc - zf + Δzᶜᶜᶜ(i, j, k-1, ibg) / 2

    Δz = ifelse(just_above_bottom, partial_Δz, full_Δz)

    return Δz
end

@inline Δzᶠᶜᶜ(i, j, k, ibg::PCIBG) = min(Δzᶜᶜᶜ(i-1, j, k, ibg), Δzᶜᶜᶜ(i, j, k, ibg))
@inline Δzᶜᶠᶜ(i, j, k, ibg::PCIBG) = min(Δzᶜᶜᶜ(i, j-1, k, ibg), Δzᶜᶜᶜ(i, j, k, ibg))
@inline Δzᶠᶠᶜ(i, j, k, ibg::PCIBG) = min(Δzᶠᶜᶜ(i, j-1, k, ibg), Δzᶠᶜᶜ(i, j, k, ibg))
      
@inline Δzᶠᶜᶠ(i, j, k, ibg::PCIBG) = min(Δzᶜᶜᶠ(i-1, j, k, ibg), Δzᶜᶜᶠ(i, j, k, ibg))
@inline Δzᶜᶠᶠ(i, j, k, ibg::PCIBG) = min(Δzᶜᶜᶠ(i, j-1, k, ibg), Δzᶜᶜᶠ(i, j, k, ibg))      
@inline Δzᶠᶠᶠ(i, j, k, ibg::PCIBG) = min(Δzᶠᶜᶠ(i, j-1, k, ibg), Δzᶠᶜᶠ(i, j, k, ibg))
