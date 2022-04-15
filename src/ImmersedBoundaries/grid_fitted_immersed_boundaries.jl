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

abstract type AbstractGridFittedBoundary <: AbstractImmersedBoundary end

const GFIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:AbstractGridFittedBoundary}

#####
##### GridFittedBottom (simple 2D immersed boundary)
#####

"""
    GridFittedBottom(bottom)

Return an immersed boundary with an irregular bottom fit to the underlying grid.
"""
struct GridFittedBottom{B} <: AbstractGridFittedBoundary
    bottom :: B
end

const OffsetArrayGridFittedBottom = GridFittedBottom{<:OffsetArray}

function ImmersedBoundaryGrid(grid, ib::OffsetArrayGridFittedBottom)
    TX, TY, TZ = topology(grid)
    return ImmersedBoundaryGrid{TX, TY, TZ}(grid, ib)
end

"""
    ImmersedBoundaryGrid(grid, ib::GridFittedBottom)

Return a grid with `GridFittedBottom` immersed boundary.

Computes ib.bottom and wraps in an array.
"""
function ImmersedBoundaryGrid(grid, ib::GridFittedBottom)
    arch = grid.architecture
    bottom_field = Field{Center, Center, Nothing}(grid)
    set!(bottom_field, ib.bottom)
    fill_halo_regions!(bottom_field)
    offset_bottom_array = dropdims(bottom_field.data, dims=3)
    new_ib = GridFittedBottom(offset_bottom_array)
    return ImmersedBoundaryGrid(grid, new_ib)
end

@inline function is_immersed(i, j, k, underlying_grid, ib::GridFittedBottom)
    x, y, z = node(c, c, c, i, j, k, underlying_grid)
    return @inbounds z < ib.bottom[i, j]
end

on_architecture(arch, ib::GridFittedBottom) = GridFittedBottom(arch_array(arch, ib.bottom))
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
                $immersed_func(i, j, k, ibg::GFIBG, closure, K, id, LX, LY, LZ, clock, Δt, κz)

        @inline $func(i, j, k, ibg::GFIBG, closure, K, id, LX, LY, LZ::Center, clock, Δt, κz) =
                $immersed_func(i, j, k, ibg::GFIBG, closure, K, id, LX, LY, LZ, clock, Δt, κz)

        @inline function $immersed_func(i, j, k, ibg::GFIBG, closure, K, id, LX, LY, LZ, clock, Δt, κz)
            return ifelse(immersed_ivd_solid_interface(LX, LY, LZ, i, j, k, ibg),
                          zero(eltype(ibg.grid)),
                          $func(i, j, k, ibg.grid, closure, K, id, LX, LY, LZ, clock, Δt, κz))
        end
    end
end

#####
##### GridFittedBoundary (experimental 3D immersed boundary)
#####

struct GridFittedBoundary{S} <: AbstractGridFittedBoundary
    mask :: S
end

function ImmersedBoundaryGrid(grid, ib::GridFittedBoundary)
    TX, TY, TZ = topology(grid)
    return ImmersedBoundaryGrid{TX, TY, TZ}(grid, ib)
end

function on_architecture(arch, b::GridFittedBoundary)
    mask = b.mask isa AbstractArray ? arch_array(arch, b.mask) : b.mask
    return GridFittedBoundary(mask)
end

@inline is_immersed(i, j, k, underlying_grid, ib::GridFittedBoundary) =
    ib.mask(node(c, c, c, i, j, k, underlying_grid)...)

######
###### Flux divergences.
######
#
# args... = v_immersed_bc, closure, diffusivities, velocities, tracers, clock, buoyancy)
#
#

immersed_τ₁ⱼ_east(i, j, k, grid, args...) = zero(eltype(grid))
immersed_τ₁ⱼ_east(i, j, k, grid, u_immersed_bc::FBC, clock, model_fields, closure) = ubc.condition(i, j, k, grid, clock...)

@inline function immersed_∂ⱼ_τ₁ⱼ(i, j, k, grid::GFIBG, u_bc, clock, model_fields, closure)
    east_flux   = immersed_τ₁ⱼ_east(i, j, k, args...)
    west_flux   = immersed_τ₁ⱼ_west(i, j, k, args...)
    south_flux  = immersed_τ₁ⱼ_south(i, j, k, args...)
    north_flux  = immersed_τ₁ⱼ_north(i, j, k, args...)
    top_flux    = immersed_τ₁ⱼ_top(i, j, k, args...)
    bottom_flux = immersed_τ₁ⱼ_bottom(i, j, k, args...)
    return 0
end

@inline function immersed_∂ⱼ_τ₂ⱼ(i, j, k, grid::GFIBG, args...)
    return 0
end

@inline function immersed_∂ⱼ_τ₃ⱼ(i, j, k, grid::GFIBG, args...)
    return 0
end

@inline function immresed_∇_dot_qᶜ(i, j, k, grid::GFIBG, args...)

    return 0
end

