using Adapt
using CUDA: CuArray
using Oceananigans.Fields: fill_halo_regions!
using Oceananigans.Architectures: arch_array

import Oceananigans.TurbulenceClosures: ivd_upper_diagonal,
                                        ivd_lower_diagonal

#####
##### PartialCellBottom
#####

"""
    PartialCellBottom(bottom, minimum_height)

Return an immersed boundary...
"""

struct PartialCellBottom{B, E} <: AbstractGridFittedBoundary
    bottom_height :: B
    minimum_fractional_partial_Δz :: E
end

@inline get_bottom_height(i, j, k, grid, bottom_height::AbstractArray) = @inbounds bottom_height[i, j]

@inline function get_bottom_height(i, j, k, underlying_grid, bottom_height)
    x, y, z = node(c, c, f, i, j, k, underlying_grid)
    return bottom_height(x, y)
end

"""

        --x--
          ∘   k+1
    k+1 --x--    ↑      <- node z
          ∘   k  | Δz
      k --x--    ↓
      
Criterion is h >= z - ϵ Δz

"""
@inline function is_immersed(i, j, k, underlying_grid, ib::PartialCellBottom)
    # Face node above current cell
    x, y, z = node(c, c, f, i, j, k+1, underlying_grid)
    Δz = Δzᶜᶜᶜ(i, j, k, underlying_grid)
    ϵ = ib.minimum_fractional_partial_Δz
    h = get_bottom_height(i, j, k, underlying_grid, ib.bottom_height)
    return h >= z - ϵ * Δz
end

const PCIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:PartialCellBottom}

bottom_cell(i, j, k, ibg::PCIBG) = !is_immersed(i, j, k, ibg.grid, ibg.immersed_boundary) & is_immersed(i, j, k-1, ibg.grid, ibg.immersed_boundary)

@inline function Δzᶜᶜᶜ(i, j, k, ibg::PCIBG)
    underlying_grid = ibg.grid
    ib = ibg.immersed_boundary
    # Get node at face above
    x, y, z = node(c, c, c, i, j, k+1, underlying_grid)

    # Get bottom height and fractional Δz parameter
    h = get_bottom_height(i, j, k, underlying_grid, ib.bottom_height)
    ϵ = ibg.immersed_boundary.minimum_fractional_partial_Δz

    # Are we in a bottom cell?
    at_the_bottom = bottom_cell(i, j, k, ibg)

    full_Δz = Δzᶜᶜᶜ(i, j, k, ibg.grid)
    partial_Δz = max(ϵ * full_Δz, z - h)

    return ifelse(at_the_bottom, partial_Δz, full_Δz)
end

@inline Δzᶠᶜᶜ(i, j, k, ibg::PCIBG) = min(Δzᶜᶜᶜ(i-1, j, k, ibg), Δzᶜᶜᶜ(i, j, k, ibg))
@inline Δzᶜᶠᶜ(i, j, k, ibg::PCIBG) = min(Δzᶜᶜᶜ(i, j-1, k, ibg), Δzᶜᶜᶜ(i, j, k, ibg))

# TODO define Δzᶜᶜᶠ
