abstract type AbstractGridFittedBoundary <: AbstractImmersedBoundary end

import Oceananigans.TurbulenceClosures: ivd_upper_diagonal, ivd_lower_diagonal

const GFIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:AbstractGridFittedBoundary}

#####
##### Implicit vertical diffusion
#####
##### For a center solver we have to check the interface "solidity" at faces k+1 in both the
##### Upper diagonal and the Lower diagonal 
##### (because of tridiagonal convention where lower_diagonal on row k is found at k-1)
##### Same goes for the face solver, where we check at centers k in both Upper and lower diagonal
#####

@inline immersed_ivd_peripheral_node(i, j, k, ibg, LX, LY, ::Center) = immersed_peripheral_node(i, j, k+1, ibg, LX, LY, Face())
@inline immersed_ivd_peripheral_node(i, j, k, ibg, LX, LY, ::Face)   = immersed_peripheral_node(i, j, k,   ibg, LX, LY, Center())

# Extend the upper and lower diagonal functions of the batched tridiagonal solver

for location in (:upper_, :lower_)
    ordinary_func = Symbol(:ivd_ ,         location, :diagonal)
    immersed_func = Symbol(:immersed_ivd_, location, :diagonal)
    @eval begin
        # Disambiguation
        @inline $ordinary_func(i, j, k, ibg::IBG, closure, K, id, ℓx, ℓy, ℓz::Face, clock, Δt, κz) =
                $immersed_func(i, j, k, ibg::IBG, closure, K, id, ℓx, ℓy, ℓz, clock, Δt, κz)

        @inline $ordinary_func(i, j, k, ibg::IBG, closure, K, id, ℓx, ℓy, ℓz::Center, clock, Δt, κz) =
                $immersed_func(i, j, k, ibg::IBG, closure, K, id, ℓx, ℓy, ℓz, clock, Δt, κz)

        @inline $immersed_func(i, j, k, ibg::IBG, closure, K, id, ℓx, ℓy, ℓz, clock, Δt, κz) =
            ifelse(immersed_ivd_peripheral_node(i, j, k, ibg, ℓx, ℓy, ℓz),
                   zero(ibg),
                   $ordinary_func(i, j, k, ibg.underlying_grid, closure, K, id, ℓx, ℓy, ℓz, clock, Δt, κz))
    end
end

# Support for Flat grids
# Note that instances of AbstractGridFittedBoundary should define _immersed_cell
# rather than immersed_cell.
const AGFB = AbstractGridFittedBoundary

@inline immersed_cell(i, j, k, grid, ib) = _immersed_cell(i, j, k, grid, ib)

@eval begin
    @inline immersed_cell(i, j, k, grid::AbstractGrid{<:Any, Flat, <:Any, <:Any}, ib::AGFB) = _immersed_cell(1, j, k, grid, ib)
    @inline immersed_cell(i, j, k, grid::AbstractGrid{<:Any, <:Any, Flat, <:Any}, ib::AGFB) = _immersed_cell(i, 1, k, grid, ib)
    @inline immersed_cell(i, j, k, grid::AbstractGrid{<:Any, <:Any, <:Any, Flat}, ib::AGFB) = _immersed_cell(i, j, 1, grid, ib)
    @inline immersed_cell(i, j, k, grid::AbstractGrid{<:Any, Flat, Flat, <:Any},  ib::AGFB) = _immersed_cell(1, 1, k, grid, ib)
    @inline immersed_cell(i, j, k, grid::AbstractGrid{<:Any, Flat, <:Any, Flat},  ib::AGFB) = _immersed_cell(1, j, 1, grid, ib)
    @inline immersed_cell(i, j, k, grid::AbstractGrid{<:Any, <:Any, Flat, Flat},  ib::AGFB) = _immersed_cell(i, 1, 1, grid, ib)
    @inline immersed_cell(i, j, k, grid::AbstractGrid{<:Any, Flat, Flat, Flat},   ib::AGFB) = _immersed_cell(1, 1, 1, grid, ib)
end

