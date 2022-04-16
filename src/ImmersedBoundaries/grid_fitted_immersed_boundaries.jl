using Adapt
using CUDA: CuArray
using OffsetArrays: OffsetArray
using Oceananigans.Fields: fill_halo_regions!
using Oceananigans.Architectures: arch_array
using Oceananigans.BoundaryConditions: FBC

import Oceananigans.TurbulenceClosures: ivd_upper_diagonal,
                                        ivd_lower_diagonal

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
##### GridFittedBottom (2.5D immersed boundary with modified bottom height)
#####

"""
    GridFittedBottom(bottom)

Return an immersed boundary with an irregular bottom fit to the underlying grid.
"""
struct GridFittedBottom{B} <: AbstractGridFittedBoundary
    bottom_height :: B
end

"""
    ImmersedBoundaryGrid(grid, ib::GridFittedBottom)

Return a grid with `GridFittedBottom` immersed boundary.

Computes ib.bottom_height and wraps in an array.
"""
function ImmersedBoundaryGrid(grid, ib::GridFittedBottom)
    arch = grid.architecture
    bottom_field = Field{Center, Center, Nothing}(grid)
    set!(bottom_field, ib.bottom_height)
    fill_halo_regions!(bottom_field)
    offset_bottom_array = dropdims(bottom_field.data, dims=3)
    new_ib = GridFittedBottom(offset_bottom_array)
    return ImmersedBoundaryGrid(grid, new_ib)
end

function ImmersedBoundaryGrid(grid, ib::GridFittedBottom{<:AbstractArray})
    TX, TY, TZ = topology(grid)
    return ImmersedBoundaryGrid{TX, TY, TZ}(grid, ib)
end

@inline function immersed_cell(i, j, k, underlying_grid, ib::GridFittedBottom)
    z = znode(c, c, c, i, j, k, underlying_grid)
    return @inbounds z < ib.bottom_height[i, j]
end

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

@inline immersed_ivd_solid_interface(LX, LY, ::Center, i, j, k, ibg) = solid_interface(LX, LY, Face(), i, j, k+1, ibg)
@inline immersed_ivd_solid_interface(LX, LY, ::Face, i, j, k, ibg)   = solid_interface(LX, LY, Center(), i, j, k, ibg)

# extending the upper and lower diagonal functions of the batched tridiagonal solver

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
            return ifelse(immersed_ivd_solid_interface(ℓx, ℓy, ℓz, i, j, k, ibg),
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

@inline immersed_cell(i, j, k, underlying_grid, ib::GridFittedBoundary{<:AbstractArray}) = @inbounds ib.mask[i, j, k]

@inline function immersed_cell(i, j, k, underlying_grid, ib::GridFittedBoundary)
    x, y, z = node(c, c, c, i, j, k, underlying_grid)
    return ib.mask(x, y, z)
end

function compute_mask(grid, ib)
    arch = grid.architecture
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

on_architecture(arch, ib::GridFittedBoundary{<:AbstractArray}) = GridFittedBoundary(arch_array(arch, ib.mask))
on_architecture(arch, ib::GridFittedBoundary{<:Field}) = GridFittedBoundary(compute_mask(on_architecture(arch, ib.mask.grid), ib))
on_architecture(arch, ib::GridFittedBoundary) = ib # need a workaround...

Adapt.adapt_structure(to, ib::GridFittedBoundary) = GridFittedBoundary(adapt(to, ib.mask))

