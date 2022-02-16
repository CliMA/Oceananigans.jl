
const c = Center()
const f = Face()

const XBoundedGrid = Union{AbstractGrid{<:Any, <:Bounded}, 
                           AbstractGrid{<:Any, <:Bounded, <:Bounded}, 
                           AbstractGrid{<:Any, <:Bounded, <:Bounded, <:Bounded}}

const YBoundedGrid = Union{AbstractGrid{<:Any, <:Any, <:Bounded},
                           AbstractGrid{<:Any, <:Any, <:Bounded, <:Bounded}}

const ZBoundedGrid = AbstractGrid{<:Any, <:Any, <:Any, <:Bounded}

# Fallback for general grids
@inline solid_node(i, j, k, grid) = false
@inline solid_node(i, j, k, grid::XBoundedGrid) = ifelse((i < 1) | (i > grid.Nx), true, false)
@inline solid_node(i, j, k, grid::YBoundedGrid) = ifelse((j < 1) | (j > grid.Ny), true, false)
@inline solid_node(i, j, k, grid::ZBoundedGrid) = ifelse((k < 1) | (k > grid.Nz), true, false)

@inline solid_node(LX, LY, LZ, i, j, k, grid)      = solid_node(i, j, k, grid)
@inline solid_interface(LX, LY, LZ, i, j, k, grid) = solid_node(i, j, k, grid)

@inline solid_node(::Face, LY, LZ, i, j, k, grid) = solid_node(i, j, k, grid) & solid_node(i-1, j, k, grid)
@inline solid_node(LX, ::Face, LZ, i, j, k, grid) = solid_node(i, j, k, grid) & solid_node(i, j-1, k, grid)
@inline solid_node(LX, LY, ::Face, i, j, k, grid) = solid_node(i, j, k, grid) & solid_node(i, j, k-1, grid)

@inline solid_node(::Face, ::Face, LZ, i, j, k, grid) = solid_node(c, f, c, i, j, k, grid) & solid_node(c, f, c, i-1, j, k, grid)
@inline solid_node(::Face, LY, ::Face, i, j, k, grid) = solid_node(c, c, f, i, j, k, grid) & solid_node(c, c, f, i-1, j, k, grid)
@inline solid_node(LX, ::Face, ::Face, i, j, k, grid) = solid_node(c, f, c, i, j, k, grid) & solid_node(c, f, c, i, j, k-1, grid)

@inline solid_node(::Face, ::Face, ::Face, i, j, k, grid) = solid_node(c, f, f, i, j, k, grid) & solid_node(c, f, f, i-1, j, k, grid)

@inline solid_interface(::Face, LY, LZ, i, j, k, grid) = solid_node(i, j, k, grid) | solid_node(i-1, j, k, grid)
@inline solid_interface(LX, ::Face, LZ, i, j, k, grid) = solid_node(i, j, k, grid) | solid_node(i, j-1, k, grid)
@inline solid_interface(LX, LY, ::Face, i, j, k, grid) = solid_node(i, j, k, grid) | solid_node(i, j, k-1, grid)

@inline solid_interface(::Face, ::Face, LZ, i, j, k, grid) = solid_interface(c, f, c, i, j, k, grid) | solid_interface(c, f, c, i-1, j, k, grid)
@inline solid_interface(::Face, LY, ::Face, i, j, k, grid) = solid_interface(c, c, f, i, j, k, grid) | solid_interface(c, c, f, i-1, j, k, grid)
@inline solid_interface(LX, ::Face, ::Face, i, j, k, grid) = solid_interface(c, f, c, i, j, k, grid) | solid_interface(c, f, c, i, j, k-1, grid)

@inline solid_interface(::Face, ::Face, ::Face, i, j, k, grid) = solid_interface(c, f, f, i, j, k, grid) | solid_interface(c, f, f, i-1, j, k, grid)
