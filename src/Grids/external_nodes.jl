
const c = Center()
const f = Face()

const XBoundedGrid = AbstractGrid{<:Any, <:Bounded}
const YBoundedGrid = AbstractGrid{<:Any, <:Any, <:Bounded}
const ZBoundedGrid = AbstractGrid{<:Any, <:Any, <:Any, <:Bounded}

const XYBoundedGrid = AbstractGrid{<:Any, <:Bounded, <:Bounded}
const XZBoundedGrid = AbstractGrid{<:Any, <:Bounded, <:Any, <:Bounded}
const YZBoundedGrid = AbstractGrid{<:Any, <:Any, <:Bounded, <:Bounded}

const FullBoundedGrid =  AbstractGrid{<:Any, <:Bounded, <:Bounded, <:Bounded}

# Fallback for general grids
@inline external_node(i, j, k, grid) = false
@inline external_node(i, j, k, grid::XBoundedGrid) = (i < 1) | (i > grid.Nx)
@inline external_node(i, j, k, grid::YBoundedGrid) = (j < 1) | (j > grid.Ny)
@inline external_node(i, j, k, grid::ZBoundedGrid) = (k < 1) | (k > grid.Nz)

@inline external_node(i, j, k, grid::XYBoundedGrid) = (i < 1) | (i > grid.Nx) | (j < 1) | (j > grid.Ny)
@inline external_node(i, j, k, grid::XZBoundedGrid) = (i < 1) | (i > grid.Nx) | (k < 1) | (k > grid.Nz)
@inline external_node(i, j, k, grid::YZBoundedGrid) = (j < 1) | (j > grid.Ny) | (k < 1) | (k > grid.Nz)

@inline external_node(i, j, k, grid::FullBoundedGrid) = (i < 1) | (i > grid.Nx) |
                                                        (j < 1) | (j > grid.Ny) |
                                                        (k < 1) | (k > grid.Nz)

@inline external_node(LX, LY, LZ, i, j, k, grid) = external_node(i, j, k, grid)
@inline peripheral_node(LX, LY, LZ, i, j, k, grid) = external_node(i, j, k, grid)

@inline external_node(::Face, LY, LZ, i, j, k, grid) = external_node(i, j, k, grid) & external_node(i-1, j, k, grid)
@inline external_node(LX, ::Face, LZ, i, j, k, grid) = external_node(i, j, k, grid) & external_node(i, j-1, k, grid)
@inline external_node(LX, LY, ::Face, i, j, k, grid) = external_node(i, j, k, grid) & external_node(i, j, k-1, grid)

@inline external_node(::Face, ::Face, LZ, i, j, k, grid) = external_node(c, f, c, i, j, k, grid) & external_node(c, f, c, i-1, j, k, grid)
@inline external_node(::Face, LY, ::Face, i, j, k, grid) = external_node(c, c, f, i, j, k, grid) & external_node(c, c, f, i-1, j, k, grid)
@inline external_node(LX, ::Face, ::Face, i, j, k, grid) = external_node(c, f, c, i, j, k, grid) & external_node(c, f, c, i, j, k-1, grid)

@inline external_node(::Face, ::Face, ::Face, i, j, k, grid) = external_node(c, f, f, i, j, k, grid) & external_node(c, f, f, i-1, j, k, grid)

@inline peripheral_node(::Face, LY, LZ, i, j, k, grid) = external_node(i, j, k, grid) | external_node(i-1, j, k, grid)
@inline peripheral_node(LX, ::Face, LZ, i, j, k, grid) = external_node(i, j, k, grid) | external_node(i, j-1, k, grid)
@inline peripheral_node(LX, LY, ::Face, i, j, k, grid) = external_node(i, j, k, grid) | external_node(i, j, k-1, grid)

@inline peripheral_node(::Face, ::Face, LZ, i, j, k, grid) = peripheral_node(c, f, c, i, j, k, grid) | peripheral_node(c, f, c, i-1, j, k, grid)
@inline peripheral_node(::Face, LY, ::Face, i, j, k, grid) = peripheral_node(c, c, f, i, j, k, grid) | peripheral_node(c, c, f, i-1, j, k, grid)
@inline peripheral_node(LX, ::Face, ::Face, i, j, k, grid) = peripheral_node(c, f, c, i, j, k, grid) | peripheral_node(c, f, c, i, j, k-1, grid)

@inline peripheral_node(::Face, ::Face, ::Face, i, j, k, grid) = peripheral_node(c, f, f, i, j, k, grid) | peripheral_node(c, f, f, i-1, j, k, grid)

