using Adapt
using CUDA: CuArray
using OffsetArrays: OffsetArray
using Oceananigans.Utils: getnamewrapper
using Oceananigans.Fields: fill_halo_regions!
using Oceananigans.Architectures: arch_array
using Oceananigans.BoundaryConditions: FBC
using Printf

import Oceananigans.TurbulenceClosures: ivd_upper_diagonal,
                                        ivd_lower_diagonal,
                                        bottom

import Oceananigans.TurbulenceClosures: immersed_∂ⱼ_τ₁ⱼ,
                                        immersed_∂ⱼ_τ₂ⱼ,
                                        immersed_∂ⱼ_τ₃ⱼ,
                                        immersed_∇_dot_qᶜ

#####
##### Some conveniences for grid fitted boundaries
#####

abstract type AbstractGridFittedBoundary <: AbstractImmersedBoundary end

const GFIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:AbstractGridFittedBoundary}

#####
##### ImmersedBoundaryGrids require one additional halo to check `inactive_node` for
##### Faces on the first halo
#####

#####
##### GridFittedBottom (2.5D immersed boundary with modified bottom height)
#####

abstract type AbstractGridFittedBottom{H} <: AbstractGridFittedBoundary end

struct CenterImmersedCondition end
struct InterfaceImmersedCondition end

"""
    GridFittedBottom(bottom_height, [immersed_condition=CenterImmersedCondition()])

Return an immersed boundary with an irregular bottom fit to the underlying grid.
"""
struct GridFittedBottom{H, I} <: AbstractGridFittedBottom{H}
    bottom_height :: H
    immersed_condition :: I
end

GridFittedBottom(bottom_height) = GridFittedBottom(bottom_height, CenterImmersedCondition())

function Base.summary(ib::GridFittedBottom)
    hmax = maximum(parent(ib.bottom_height))
    hmin = minimum(parent(ib.bottom_height))
    return @sprintf("GridFittedBottom(min(h)=%.2e, max(h)=%.2e)", hmin, hmax)
end

Base.summary(ib::GridFittedBottom{<:Function}) = @sprintf("GridFittedBottom(%s)", ib.bottom_height)

Base.show(io::IO, ib::GridFittedBottom) = print(io, summary(ib))

"""
    ImmersedBoundaryGrid(grid, ib::GridFittedBottom)

Return a grid with `GridFittedBottom` immersed boundary.

Computes ib.bottom_height and wraps in an array.
"""
function ImmersedBoundaryGrid(grid, ib::AbstractGridFittedBottom)
    bottom_field = Field((Center, Center, Nothing), grid)
    set!(bottom_field, ib.bottom_height)
    fill_halo_regions!(bottom_field)
    offset_bottom_array = dropdims(bottom_field.data, dims=3)

    # TODO: maybe clean this up
    new_ib = getnamewrapper(ib)(offset_bottom_array)

    return ImmersedBoundaryGrid(grid, new_ib)
end

function ImmersedBoundaryGrid(grid, ib::AbstractGridFittedBottom{<:OffsetArray})
    TX, TY, TZ = topology(grid)
    validate_ib_size(grid, ib)
    return ImmersedBoundaryGrid{TX, TY, TZ}(grid, ib)
end

function validate_ib_size(grid, ib)
    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)

    bottom_height_size = (Nx, Ny) .+ 2 .* (Hx, Hy)

    # Check that the size of a bottom field are 
    # consistent with the size of the field
    any(size(ib.bottom_height) .!= bottom_height_size) && 
        throw(ArgumentError("The dimensions of the immersed boundary $(size(ib.bottom_height)) do not match the grid size $(bottom_height_size)"))
end

@inline function _immersed_cell(i, j, k, underlying_grid, ib::GridFittedBottom{<:Any, <:InterfaceImmersedCondition})
    z = znode(i, j, k+1, underlying_grid, c, c, f)
    h = @inbounds ib.bottom_height[i, j]
    return z <= h
end

@inline function _immersed_cell(i, j, k, underlying_grid, ib::GridFittedBottom{<:Any, <:CenterImmersedCondition})
    z = znode(i, j, k, underlying_grid, c, c, c)
    h = @inbounds ib.bottom_height[i, j]
    return z <= h
end

@inline bottom(i, j, k, ibg::GFIBG) = @inbounds ibg.immersed_boundary.bottom_height[i, j]

on_architecture(arch, ib::GridFittedBottom) = GridFittedBottom(arch_array(arch, ib.bottom_height))
Adapt.adapt_structure(to, ib::GridFittedBottom) = GridFittedBottom(adapt(to, ib.bottom_height))     

#####
##### Implicit vertical diffusion
#####

####
#### For a center solver we have to check the interface "solidity" at faces k+1 in both the
#### Upper diagonal and the Lower diagonal 
#### (because of tridiagonal convention where lower_diagonal on row k is found at k-1)
#### Same goes for the face solver, where we check at centers k in both Upper and lower diagonal
####

@inline immersed_ivd_peripheral_node(i, j, k, ibg, LX, LY, ::Center) = immersed_peripheral_node(i, j, k+1, ibg, LX, LY, Face())
@inline immersed_ivd_peripheral_node(i, j, k, ibg, LX, LY, ::Face)   = immersed_peripheral_node(i, j, k,   ibg, LX, LY, Center())

# Extend the upper and lower diagonal functions of the batched tridiagonal solver

for location in (:upper_, :lower_)
    immersed_func = Symbol(:immersed_ivd_, location, :diagonal)
    ordinary_func = Symbol(:ivd_ ,         location, :diagonal)
    @eval begin
        # Disambiguation
        @inline $ordinary_func(i, j, k, ibg::GFIBG, closure, K, id, ℓx, ℓy, ℓz::Face, clock, Δt, κz) =
                $immersed_func(i, j, k, ibg::GFIBG, closure, K, id, ℓx, ℓy, ℓz, clock, Δt, κz)

        @inline $ordinary_func(i, j, k, ibg::GFIBG, closure, K, id, ℓx, ℓy, ℓz::Center, clock, Δt, κz) =
                $immersed_func(i, j, k, ibg::GFIBG, closure, K, id, ℓx, ℓy, ℓz, clock, Δt, κz)

        @inline function $immersed_func(i, j, k, ibg::GFIBG, closure, K, id, ℓx, ℓy, ℓz, clock, Δt, κz)
            return ifelse(immersed_ivd_peripheral_node(i, j, k, ibg, ℓx, ℓy, ℓz),
                          zero(eltype(ibg.underlying_grid)),
                          $ordinary_func(i, j, k, ibg.underlying_grid, closure, K, id, ℓx, ℓy, ℓz, clock, Δt, κz))
        end
    end
end

#####
##### GridFittedBoundary (experimental 3D immersed boundary)
#####

struct GridFittedBoundary{M} <: AbstractGridFittedBoundary
    mask :: M
end

@inline _immersed_cell(i, j, k, underlying_grid, ib::GridFittedBoundary{<:AbstractArray}) = @inbounds ib.mask[i, j, k]

@inline function _immersed_cell(i, j, k, underlying_grid, ib::GridFittedBoundary)
    x, y, z = node(i, j, k, underlying_grid, c, c, c)
    return ib.mask(x, y, z)
end

function compute_mask(grid, ib)
    mask_field = Field{Center, Center, Center}(grid, Bool)
    set!(mask_field, ib.mask)
    fill_halo_regions!(mask_field)
    return mask_field
end

function ImmersedBoundaryGrid(grid, ib::GridFittedBoundary; precompute_mask=true)
    TX, TY, TZ = topology(grid)

    if precompute_mask
        mask_field = compute_mask(grid, ib)
        new_ib = GridFittedBoundary(mask_field)
        return ImmersedBoundaryGrid{TX, TY, TZ}(grid, new_ib)
    else
        return ImmersedBoundaryGrid{TX, TY, TZ}(grid, ib)
    end
end

function ImmersedBoundaryGrid(grid, ib::GridFittedBoundary{<:OffsetArray}; kw...)
    TX, TY, TZ = topology(grid)
    return ImmersedBoundaryGrid{TX, TY, TZ}(grid, ib)
end

on_architecture(arch, ib::GridFittedBoundary{<:AbstractArray}) = GridFittedBoundary(arch_array(arch, ib.mask))
on_architecture(arch, ib::GridFittedBoundary{<:Field}) = GridFittedBoundary(compute_mask(on_architecture(arch, ib.mask.grid), ib))
on_architecture(arch, ib::GridFittedBoundary) = ib # need a workaround...

Adapt.adapt_structure(to, ib::AbstractGridFittedBoundary) = GridFittedBoundary(adapt(to, ib.mask))

# fallback
immersed_cell(i, j, k, grid, ib) = _immersed_cell(i, j, k, grid, ib)

# support for Flat grids
using Oceananigans.Grids: AbstractGrid
for ImmBoundary in [:GridFittedBottom, :GridFittedBoundary]
    @eval begin
        @inline immersed_cell(i, j, k, grid::AbstractGrid{<:Any, Flat, <:Any, <:Any}, ib::$ImmBoundary) = _immersed_cell(1, j, k, grid, ib)
        @inline immersed_cell(i, j, k, grid::AbstractGrid{<:Any, <:Any, Flat, <:Any}, ib::$ImmBoundary) = _immersed_cell(i, 1, k, grid, ib)
        @inline immersed_cell(i, j, k, grid::AbstractGrid{<:Any, <:Any, <:Any, Flat}, ib::$ImmBoundary) = _immersed_cell(i, j, 1, grid, ib)
        @inline immersed_cell(i, j, k, grid::AbstractGrid{<:Any, Flat, Flat, <:Any},  ib::$ImmBoundary) = _immersed_cell(1, 1, k, grid, ib)
        @inline immersed_cell(i, j, k, grid::AbstractGrid{<:Any, Flat, <:Any, Flat},  ib::$ImmBoundary) = _immersed_cell(1, j, 1, grid, ib)
        @inline immersed_cell(i, j, k, grid::AbstractGrid{<:Any, <:Any, Flat, Flat},  ib::$ImmBoundary) = _immersed_cell(i, 1, 1, grid, ib)
        @inline immersed_cell(i, j, k, grid::AbstractGrid{<:Any, Flat, Flat, Flat},   ib::$ImmBoundary) = _immersed_cell(1, 1, 1, grid, ib)
    end
end

