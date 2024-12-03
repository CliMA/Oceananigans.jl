abstract type AbstractGridFittedBoundary <: AbstractImmersedBoundary end

const GFIBG = ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:AbstractGridFittedBoundary}

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

function clamp_bottom_height!(bottom_field, grid)
    launch!(architecture(grid), grid, :xy, _clamp_bottom_height!, bottom_field, grid)
    return nothing
end

const c = Center()
const f = Face()

@kernel function _clamp_bottom_height!(z, grid)
    i, j = @index(Global, NTuple)
    Nz = size(grid, 3)
    zmin = znode(i, j, 1,    grid, c, c, f)
    zmax = znode(i, j, Nz+1, grid, c, c, f)
    @inbounds z[i, j, 1] = clamp(z[i, j, 1], zmin, zmax)
end


