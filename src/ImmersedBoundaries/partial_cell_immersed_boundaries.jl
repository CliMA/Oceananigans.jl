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

@inline function Δzᵃᵃᶜ(i, j, k, ibg::PCIBG)
    x, y, z = node(c, c, c, i, j, k, underlying_grid)
    Δzᵃᵃᶜ = ibg.immersed_boundary.bottom_height(x, y)
    return Δzᵃᵃᶜ
end

#height = (topography.(grid.yᵃᶠᵃ[1:grid.Ny]) + 4*topography.(grid.yᵃᶜᵃ[1:grid.Ny]) + topography.(grid.yᵃᶠᵃ[2:grid.Ny+1]))/6

# FJP: Need to add a second argument???
const ArrayPartialCellBottom = PartialCellBottom{<:Array}
const CuArrayPartialCellBottom = PartialCellBottom{<:CuArray}

function ImmersedBoundaryGrid(grid, ib::Union{ArrayPartialCellBottom, CuArrayPartialCellBottom})

    # Wrap bathymetry in an OffsetArray with halos
    arch = grid.architecture
    bottom_field = Field{Center, Center, Nothing}(grid)
    bottom_data = arch_array(arch, ib.bottom_height)
    bottom_field .= bottom_data
    fill_halo_regions!(bottom_field, arch)
    offset_bottom_array = dropdims(bottom_field.data, dims=3)
    new_ib = PartialCellBottom(offset_bottom_array, ib.minimum_fractional_partial_Δz)
    return ImmersedBoundaryGrid(grid, new_ib)
end
