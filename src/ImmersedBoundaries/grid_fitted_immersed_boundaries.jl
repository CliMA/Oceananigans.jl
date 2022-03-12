using Adapt
using CUDA: CuArray
using Oceananigans.Fields: fill_halo_regions!
using Oceananigans.Architectures: arch_array

import Oceananigans.TurbulenceClosures: ivd_upper_diagonal,
                                        ivd_lower_diagonal

abstract type AbstractGridFittedBoundary <: AbstractImmersedBoundary end

const GFIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:AbstractGridFittedBoundary}

#####
##### GridFittedBoundary
#####

struct GridFittedBoundary{S} <: AbstractGridFittedBoundary
    mask :: S
end

@inline is_immersed(i, j, k, underlying_grid, ib::GridFittedBoundary) = ib.mask(node(c, c, c, i, j, k, underlying_grid)...)

#####
##### GridFittedBottom
#####

"""
    GridFittedBottom(bottom)

Return an immersed boundary.
"""
struct GridFittedBottom{B} <: AbstractGridFittedBoundary
    bottom :: B
end

@inline function is_immersed(i, j, k, underlying_grid, ib::GridFittedBottom)
    x, y, z = node(c, c, c, i, j, k, underlying_grid)
    return z < ib.bottom(x, y)
end

@inline function is_immersed(i, j, k, underlying_grid, ib::GridFittedBottom{<:AbstractArray})
    x, y, z = node(c, c, c, i, j, k, underlying_grid)
    return @inbounds z < ib.bottom[i, j]
end

const ArrayGridFittedBottom = GridFittedBottom{<:Array}
const CuArrayGridFittedBottom = GridFittedBottom{<:CuArray}

function ImmersedBoundaryGrid(grid, ib::Union{ArrayGridFittedBottom, CuArrayGridFittedBottom})
    # Wrap bathymetry in an OffsetArray with halos
    arch = grid.architecture
    bottom_field = Field{Center, Center, Nothing}(grid)
    bottom_data = arch_array(arch, ib.bottom)
    bottom_field .= bottom_data
    fill_halo_regions!(bottom_field, arch)
    offset_bottom_array = dropdims(bottom_field.data, dims=3)
    new_ib = GridFittedBottom(offset_bottom_array)
    return ImmersedBoundaryGrid(grid, new_ib)
end

Adapt.adapt_structure(to, ib::GridFittedBottom) = GridFittedBottom(adapt(to, ib.bottom))     

#####
##### Implicit vertical diffusion
#####

####
#### For a center solver we have to check the interface "solidity" at faces k+1 in both the
#### Upper diagonal and the Lower diagonal 
#### (because of tridiagonal convention where lower_diagonal on row k is found at k-1)
#### Same goes for the face solver, where we check at centers k in both Upper and lower diagonal
####

@inline immersed_ivd_solid_interface(LX, LY, ::Center, i, j, k, ibg) = solid_interface(LX, LY, Face(), i, j, k+1, ibg)
@inline immersed_ivd_solid_interface(LX, LY, ::Face, i, j, k, ibg)   = solid_interface(LX, LY, Center(), i, j, k, ibg)

# extending the upper and lower diagonal functions of the batched tridiagonal solver

for location in (:upper_, :lower_)
    immersed_func = Symbol(:immersed_ivd_, location, :diagonal)
    func = Symbol(:ivd_ , location, :diagonal)
    @eval begin
        # Disambiguation
        @inline $func(i, j, k, ibg::GFIBG, closure, K, id, LX, LY, LZ::Face, clock, Δt, κz) =
                $alt_func(i, j, k, ibg::GFIBG, closure, K, id, LX, LY, LZ, clock, Δt, κz)

        @inline $func(i, j, k, ibg::GFIBG, closure, K, id, LX, LY, LZ::Center, clock, Δt, κz) =
                $alt_func(i, j, k, ibg::GFIBG, closure, K, id, LX, LY, LZ, clock, Δt, κz)

        @inline function $immersed_func(i, j, k, ibg::GFIBG, closure, K, id, LX, LY, LZ, clock, Δt, κz)
            return ifelse(immersed_ivd_solid_interface(LX, LY, LZ, i, j, k, ibg),
                          zero(eltype(ibg.grid)),
                          $func(i, j, k, ibg.grid, closure, K, id, LX, LY, LZ, clock, Δt, κz))
        end
    end
end

