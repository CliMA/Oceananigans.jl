
const c = Center()
const f = Face()

const XBoundedGrid = AbstractGrid{<:Any, <:Bounded}
const YBoundedGrid = AbstractGrid{<:Any, <:Any, <:Bounded}
const ZBoundedGrid = AbstractGrid{<:Any, <:Any, <:Any, <:Bounded}

const XYBoundedGrid = AbstractGrid{<:Any, <:Bounded, <:Bounded}
const XZBoundedGrid = AbstractGrid{<:Any, <:Bounded, <:Any, <:Bounded}
const YZBoundedGrid = AbstractGrid{<:Any, <:Any, <:Bounded, <:Bounded}

const XYZBoundedGrid =  AbstractGrid{<:Any, <:Bounded, <:Bounded, <:Bounded}

#####
##### Exterior node and peripheral node
#####

"""
    exterior_cell(i, j, k, grid)

Return `true` when the tracer cell is outside the boundaries of the domain.
Return `false` either when the tracer cell at index `i, j, k` is "wet".
In non-`Bounded` directions, `exterior_node` always returns `false`.
"""
@inline exterior_cell(i, j, k, grid) = false
@inline exterior_cell(i, j, k, grid::XBoundedGrid) = (i < 1) | (i > grid.Nx)
@inline exterior_cell(i, j, k, grid::YBoundedGrid) = (j < 1) | (j > grid.Ny)
@inline exterior_cell(i, j, k, grid::ZBoundedGrid) = (k < 1) | (k > grid.Nz)

@inline exterior_cell(i, j, k, grid::XYBoundedGrid) = (i < 1) | (i > grid.Nx) | (j < 1) | (j > grid.Ny)
@inline exterior_cell(i, j, k, grid::XZBoundedGrid) = (i < 1) | (i > grid.Nx) | (k < 1) | (k > grid.Nz)
@inline exterior_cell(i, j, k, grid::YZBoundedGrid) = (j < 1) | (j > grid.Ny) | (k < 1) | (k > grid.Nz)

@inline exterior_cell(i, j, k, grid::XYZBoundedGrid) = (i < 1) | (i > grid.Nx) |
                                                       (j < 1) | (j > grid.Ny) |
                                                       (k < 1) | (k > grid.Nz)

"""
    exterior_node(LX, LY, LZ, i, j, k, grid)

Return `true` when the location `(LX, LY, LZ)`, interpreted as _either_ a
cell interface, or the cell itself, is "entirely" exterior to a `Bounded` domain.

For `Face` locations, this means the node is surrounded by `exterior_cell`s.

If a cell interface touches a "wet" cell, it is _not_ an exterior node.
"""
@inline exterior_node(LX, LY, LZ, i, j, k, grid) = exterior_cell(i, j, k, grid)

@inline exterior_node(::Face, LY, LZ, i, j, k, grid) = exterior_cell(i, j, k, grid) & exterior_cell(i-1, j, k, grid)
@inline exterior_node(LX, ::Face, LZ, i, j, k, grid) = exterior_cell(i, j, k, grid) & exterior_cell(i, j-1, k, grid)
@inline exterior_node(LX, LY, ::Face, i, j, k, grid) = exterior_cell(i, j, k, grid) & exterior_cell(i, j, k-1, grid)

@inline exterior_node(::Face, ::Face, LZ, i, j, k, grid) = exterior_node(c, f, c, i, j, k, grid) & exterior_node(c, f, c, i-1, j, k, grid)
@inline exterior_node(::Face, LY, ::Face, i, j, k, grid) = exterior_node(c, c, f, i, j, k, grid) & exterior_node(c, c, f, i-1, j, k, grid)
@inline exterior_node(LX, ::Face, ::Face, i, j, k, grid) = exterior_node(c, f, c, i, j, k, grid) & exterior_node(c, f, c, i, j, k-1, grid)

@inline exterior_node(::Face, ::Face, ::Face, i, j, k, grid) = exterior_node(c, f, f, i, j, k, grid) & exterior_node(c, f, f, i-1, j, k, grid)

"""
    peripheral_node(LX, LY, LZ, i, j, k, grid)

Return `true` when the location `(LX, LY, LZ)`, is _either_ exterior or
lies on a boundary.
"""
@inline peripheral_node(LX, LY, LZ, i, j, k, grid) = exterior_cell(i, j, k, grid)

@inline peripheral_node(::Face, LY, LZ, i, j, k, grid) = exterior_cell(i, j, k, grid) | exterior_cell(i-1, j, k, grid)
@inline peripheral_node(LX, ::Face, LZ, i, j, k, grid) = exterior_cell(i, j, k, grid) | exterior_cell(i, j-1, k, grid)
@inline peripheral_node(LX, LY, ::Face, i, j, k, grid) = exterior_cell(i, j, k, grid) | exterior_cell(i, j, k-1, grid)

@inline peripheral_node(::Face, ::Face, LZ, i, j, k, grid) = peripheral_node(c, f, c, i, j, k, grid) | peripheral_node(c, f, c, i-1, j, k, grid)
@inline peripheral_node(::Face, LY, ::Face, i, j, k, grid) = peripheral_node(c, c, f, i, j, k, grid) | peripheral_node(c, c, f, i-1, j, k, grid)
@inline peripheral_node(LX, ::Face, ::Face, i, j, k, grid) = peripheral_node(c, f, c, i, j, k, grid) | peripheral_node(c, f, c, i, j, k-1, grid)

@inline peripheral_node(::Face, ::Face, ::Face, i, j, k, grid) = peripheral_node(c, f, f, i, j, k, grid) | peripheral_node(c, f, f, i-1, j, k, grid)

"""
    boundary_node(LX, LY, LZ, i, j, k, grid)

Return `true` when the location `(LX, LY, LZ)` lies on a boundary.
"""
@inline boundary_node(LX, LY, LZ, i, j, k, grid) = peripheral_node(LX, LY, LZ, i, j, k, grid) & !exterior_node(LX, LY, LZ, i, j, k, grid)

