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
              bottom in absolute ``z`` coordinates. This is the height of
              the bottom interface of the last ``fluid`` cell.
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

on_architecture(arch, ib::GridFittedBottom) = GridFittedBottom(on_architecture(ib.bottom_height))

function on_architecture(arch, ib::GridFittedBottom{<:Field})
    architecture(ib.bottom_height) == arch && return ib
    arch_grid = on_architecture(arch, ib.bottom_height.grid)
    new_bottom_height = Field{Center, Center, Nothing}(arch_grid)
    set!(new_bottom_height, ib.bottom_height)
    fill_halo_regions!(new_bottom_height)
    return GridFittedBottom(new_bottom_height, ib.immersed_condition)
end

Adapt.adapt_structure(to, ib::GridFittedBottom) = GridFittedBottom(adapt(to, ib.z_bottom))

"""
    ImmersedBoundaryGrid(grid, ib::GridFittedBottom)

Return a grid with `GridFittedBottom` immersed boundary (`ib`).

Computes `ib.z_bottom` and wraps it in a Field.
"""
function ImmersedBoundaryGrid(grid, ib::GridFittedBottom)
    bottom_field = Field{Center, Center, Nothing}(grid)
    set!(bottom_field, ib.z_bottom)
    @apply_regionally correct_z_bottom!(bottom_field, grid)
    fill_halo_regions!(bottom_field)
    new_ib = GridFittedBottom(bottom_field)
    TX, TY, TZ = topology(grid)
    return ImmersedBoundaryGrid{TX, TY, TZ}(grid, new_ib)
end

correct_z_bottom!(bottom_field, grid, ib) = 
    launch!(architecture(grid), grid, :xy, _correct_z_bottom!, bottom_field, grid, ib)

@kernel function _correct_z_bottom!(bottom_field, grid, ::GridFittedBottom)
    i, j = @index(grid)
    zb = @inbounds bottom_field[i, j, 1]
    for k in 1:grid.Nz
        z⁺ = znode(i, j, k+1, grid, c, c, f)
        bottom_cell = zb < z⁺
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

@inline bottom_heightᶜᶜᵃ(i, j, k, ibg::GFBIBG) = ibg.immersed_boundary.bottom_height[i, j, 1]
@inline bottom_heightᶜᶠᵃ(i, j, k, ibg::GFBIBG) = min(bottom_heightᶜᶜᵃ(i, j-1, k, ibg), bottom_heightᶜᶜᵃ(i, j, k, ibg))
@inline bottom_heightᶠᶜᵃ(i, j, k, ibg::GFBIBG) = min(bottom_heightᶜᶜᵃ(i-1, j, k, ibg), bottom_heightᶜᶜᵃ(i, j, k, ibg))
@inline bottom_heightᶠᶠᵃ(i, j, k, ibg::GFBIBG) = min(bottom_heightᶠᶜᵃ(i, j-1, k, ibg), bottom_heightᶠᶜᵃ(i, j, k, ibg))
