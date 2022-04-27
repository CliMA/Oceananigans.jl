
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
    inactive_cell(i, j, k, grid)

Return `true` when the tracer cell at `i, j, k` is lies outside the "active domain" of
the grid in `Bounded` directions. Otherwise, return `false`.
"""
@inline inactive_cell(i, j, k, grid) = false
@inline inactive_cell(i, j, k, grid::XBoundedGrid) = (i < 1) | (i > grid.Nx)
@inline inactive_cell(i, j, k, grid::YBoundedGrid) = (j < 1) | (j > grid.Ny)
@inline inactive_cell(i, j, k, grid::ZBoundedGrid) = (k < 1) | (k > grid.Nz)

@inline inactive_cell(i, j, k, grid::XYBoundedGrid) = (i < 1) | (i > grid.Nx) | (j < 1) | (j > grid.Ny)
@inline inactive_cell(i, j, k, grid::XZBoundedGrid) = (i < 1) | (i > grid.Nx) | (k < 1) | (k > grid.Nz)
@inline inactive_cell(i, j, k, grid::YZBoundedGrid) = (j < 1) | (j > grid.Ny) | (k < 1) | (k > grid.Nz)

@inline inactive_cell(i, j, k, grid::XYZBoundedGrid) = (i < 1) | (i > grid.Nx) |
                                                       (j < 1) | (j > grid.Ny) |
                                                       (k < 1) | (k > grid.Nz)

"""
    inactive_node(i, j, k, grid, LX, LY, LZ)

Return `true` when the location `(LX, LY, LZ)` is "inactive" and thus not directly
associated with an "active" cell.

For `Face` locations, this means the node is surrounded by `inactive_cell`s:
the interfaces of "active" cells are _not_ `inactive_node`.

For `Center` locations, this means the direction is `Bounded` and that the
cell or interface centered on the location is completely outside the active
region of the grid.
"""
@inline inactive_node(i, j, k, grid, LX, LY, LZ) = inactive_cell(i, j, k, grid)

@inline inactive_node(i, j, k, grid, ::Face, LY, LZ) = inactive_cell(i, j, k, grid) & inactive_cell(i-1, j, k, grid)
@inline inactive_node(i, j, k, grid, LX, ::Face, LZ) = inactive_cell(i, j, k, grid) & inactive_cell(i, j-1, k, grid)
@inline inactive_node(i, j, k, grid, LX, LY, ::Face) = inactive_cell(i, j, k, grid) & inactive_cell(i, j, k-1, grid)

@inline inactive_node(i, j, k, grid, ::Face, ::Face, LZ) = inactive_node(i, j, k, grid, c, f, c) & inactive_node(i-1, j, k, grid, c, f, c)
@inline inactive_node(i, j, k, grid, ::Face, LY, ::Face) = inactive_node(i, j, k, grid, c, c, f) & inactive_node(i-1, j, k, grid, c, c, f)
@inline inactive_node(i, j, k, grid, LX, ::Face, ::Face) = inactive_node(i, j, k, grid, c, f, c) & inactive_node(i, j, k-1, grid, c, f, c)

@inline inactive_node(i, j, k, grid, ::Face, ::Face, ::Face) = inactive_node(i, j, k, grid, c, f, f) & inactive_node(i-1, j, k, grid, c, f, f)

"""
    peripheral_node(LX, LY, LZ, i, j, k, grid)

Return `true` when the location `(LX, LY, LZ)`, is _either_ inactive or
lies on the boundary between inactive and active cells in a `Bounded` direction.
"""
@inline peripheral_node(LX, LY, LZ, i, j, k, grid) = inactive_cell(i, j, k, grid)

@inline peripheral_node(i, j, k, grid, ::Face, LY, LZ) = inactive_cell(i, j, k, grid) | inactive_cell(i-1, j, k, grid)
@inline peripheral_node(i, j, k, grid, LX, ::Face, LZ) = inactive_cell(i, j, k, grid) | inactive_cell(i, j-1, k, grid)
@inline peripheral_node(i, j, k, grid, LX, LY, ::Face) = inactive_cell(i, j, k, grid) | inactive_cell(i, j, k-1, grid)

@inline peripheral_node(i, j, k, grid, ::Face, ::Face, LZ) = peripheral_node(i, j, k, grid, c, f, c) | peripheral_node(i-1, j, k, grid, c, f, c)
@inline peripheral_node(i, j, k, grid, ::Face, LY, ::Face) = peripheral_node(i, j, k, grid, c, c, f) | peripheral_node(i-1, j, k, grid, c, c, f)
@inline peripheral_node(i, j, k, grid, LX, ::Face, ::Face) = peripheral_node(i, j, k, grid, c, f, c) | peripheral_node(i, j, k-1, grid, c, f, c)

@inline peripheral_node(i, j, k, grid, ::Face, ::Face, ::Face) = peripheral_node(i, j, k, grid, c, f, f) | peripheral_node(i-1, j, k, grid, c, f, f)

"""
    boundary_node(i, j, k, grid, LX, LY, LZ)

Return `true` when the location `(LX, LY, LZ)` lies on a boundary.
"""
@inline boundary_node(i, j, k, grid, LX, LY, LZ) = peripheral_node(i, j, k, grid, LX, LY, LZ) & !inactive_node(i, j, k, grid, LX, LY, LZ)

