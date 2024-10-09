using Adapt
using CUDA: CuArray
using OffsetArrays: OffsetArray
using Oceananigans.Utils: getnamewrapper
using Oceananigans.Grids: total_size
using Oceananigans.Fields: fill_halo_regions!
using Oceananigans.BoundaryConditions: FBC
using Printf

import Oceananigans.TurbulenceClosures: z_bottom

#####
##### GridFittedBottom (2.5D immersed boundary with modified bottom height)
#####

abstract type AbstractGridFittedBottom{H} <: AbstractGridFittedBoundary end

struct GridFittedBottom{H} <: AbstractGridFittedBottom{H}
    z_bottom :: H
end

const GFBIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:GridFittedBottom}

"""
    GridFittedBottom(z_bottom)

Return a bottom immersed boundary.

Keyword Arguments
=================

* `z_bottom`: an array or function that gives the height of the
              bottom in absolute ``z`` coordinates.
"""

function Base.summary(ib::GridFittedBottom)
    zmax  = maximum(ib.z_bottom)
    zmin  = minimum(ib.z_bottom)
    zmean = mean(ib.z_bottom)

    summary1 = "GridFittedBottom("

    summary2 = string("mean(z)=", prettysummary(zmean),
                      ", min(z)=", prettysummary(zmin),
                      ", max(z)=", prettysummary(zmax))

    summary3 = ")"

    return summary1 * summary2 * summary3
end

Base.summary(ib::GridFittedBottom{<:Function}) = @sprintf("GridFittedBottom(%s)", ib.z_bottom)

function Base.show(io::IO, ib::GridFittedBottom)
    print(io, summary(ib), '\n')
    print(io, "├── z_bottom: ", prettysummary(ib.z_bottom), '\n')
end

on_architecture(arch, ib::GridFittedBottom) = GridFittedBottom(on_architecture(ib.z_bottom))

function on_architecture(arch, ib::GridFittedBottom{<:Field})
    architecture(ib.z_bottom) == arch && return ib
    arch_grid = on_architecture(arch, ib.z_bottom.grid)
    new_z_bottom = Field{Center, Center, Nothing}(arch_grid)
    set!(new_z_bottom, ib.z_bottom)
    fill_halo_regions!(new_z_bottom)
    return GridFittedBottom(new_z_bottom)
end

Adapt.adapt_structure(to, ib::GridFittedBottom) = GridFittedBottom(adapt(to, ib.z_bottom))

"""
    ImmersedBoundaryGrid(grid, ib::GridFittedBottom)

Return a grid with `GridFittedBottom` immersed boundary (`ib`).

Computes `ib.z_bottom` and wraps it in a Field. `ib.z_bottom` is the z-coordinate of top-most interface
of the last ``immersed`` cell in the column.
"""
function ImmersedBoundaryGrid(grid, ib::GridFittedBottom)
    bottom_field = Field{Center, Center, Nothing}(grid)
    set!(bottom_field, ib.z_bottom)
    @apply_regionally correct_z_bottom!(bottom_field, grid, ib)
    fill_halo_regions!(bottom_field)
    new_ib = GridFittedBottom(bottom_field)
    TX, TY, TZ = topology(grid)
    return ImmersedBoundaryGrid{TX, TY, TZ}(grid, new_ib)
end

correct_z_bottom!(bottom_field, grid, ib) = 
    launch!(architecture(grid), grid, :xy, _correct_z_bottom!, bottom_field, grid, ib)

@kernel function _correct_z_bottom!(bottom_field, grid, ::GridFittedBottom)
    i, j = @index(Global, NTuple)
    zb = @inbounds bottom_field[i, j, 1]
    for k in 1:grid.Nz
        z⁺ = znode(i, j, k+1, grid, c, c, f)
        bottom_cell = zb ≤ z⁺
        @inbounds bottom_field[i, j, 1] = ifelse(bottom_cell, z⁺, zb)
    end
end

@inline function _immersed_cell(i, j, k, underlying_grid, ib::GridFittedBottom)
    z = znode(i, j, k+1, underlying_grid, c, c, f)
    h = @inbounds ib.z_bottom[i, j, 1]
    return z ≤ h
end

@inline z_bottom(i, j, ibg::GFBIBG) = @inbounds ibg.immersed_boundary.z_bottom[i, j, 1]

#####
##### Bottom height
#####

const AGFBIB = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:AbstractGridFittedBottom}

@inline column_heightᶜᶜᵃ(i, j, k, ibg::AGFBIB) = @inbounds znode(i, j, grid.Nz+1, ibg, c, c, f) - ibg.immersed_boundary.z_bottom[i, j, 1] 
@inline column_heightᶜᶠᵃ(i, j, k, ibg::AGFBIB) = min(column_heightᶜᶜᵃ(i, j-1, k, ibg), column_heightᶜᶜᵃ(i, j, k, ibg))
@inline column_heightᶠᶜᵃ(i, j, k, ibg::AGFBIB) = min(column_heightᶜᶜᵃ(i-1, j, k, ibg), column_heightᶜᶜᵃ(i, j, k, ibg))
@inline column_heightᶠᶠᵃ(i, j, k, ibg::AGFBIB) = min(column_heightᶠᶜᵃ(i, j-1, k, ibg), column_heightᶠᶜᵃ(i, j, k, ibg))

