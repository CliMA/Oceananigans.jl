"""
    AbstractGrid{FT, TX, TY, TZ}

Abstract supertype for grids with elements of type `FT` and topology `{TX, TY, TZ}`.
"""
abstract type AbstractGrid{FT, TX, TY, TZ, Arch} end

"""
    AbstractUnderlyingGrid{FT, TX, TY, TZ}

Abstract supertype for "primary" grids (as opposed to grids with immersed boundaries)
with elements of type `FT`, topology `{TX, TY, TZ}` and vertical coordinate `CZ`.
"""
abstract type AbstractUnderlyingGrid{FT, TX, TY, TZ, CZ, Arch} <: AbstractGrid{FT, TX, TY, TZ, Arch} end

"""
    AbstractCurvilinearGrid{FT, TX, TY, TZ}

Abstract supertype for curvilinear grids with elements of type `FT`,
topology `{TX, TY, TZ}`, and vertical coordinate `CZ`.
"""
abstract type AbstractCurvilinearGrid{FT, TX, TY, TZ, CZ, Arch} <: AbstractUnderlyingGrid{FT, TX, TY, TZ, CZ, Arch} end

"""
    AbstractHorizontallyCurvilinearGrid{FT, TX, TY, TZ}

Abstract supertype for horizontally-curvilinear grids with elements of type `FT`,
topology `{TX, TY, TZ}` and vertical coordinate `CZ`.
"""
abstract type AbstractHorizontallyCurvilinearGrid{FT, TX, TY, TZ, CZ, Arch} <: AbstractCurvilinearGrid{FT, TX, TY, TZ, CZ, Arch} end

const XFlatGrid = AbstractGrid{<:Any, Flat}
const YFlatGrid = AbstractGrid{<:Any, <:Any, Flat}
const ZFlatGrid = AbstractGrid{<:Any, <:Any, <:Any, Flat}

const XYFlatGrid = AbstractGrid{<:Any, Flat, Flat}
const XZFlatGrid = AbstractGrid{<:Any, Flat, <:Any, Flat}
const YZFlatGrid = AbstractGrid{<:Any, <:Any, Flat, Flat}

const XYZFlatGrid = AbstractGrid{<:Any, Flat, Flat, Flat}

isrectilinear(grid) = false

# Fallback
@inline get_active_column_map(::AbstractGrid) = nothing
@inline get_active_cells_map(::AbstractGrid, any_map_type) = nothing

"""
    topology(grid)

Return a tuple with the topology of the `grid` for each dimension.
"""
@inline topology(::AbstractGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} = (TX, TY, TZ)

"""
    topology(grid, dim)

Return the topology of the `grid` for the `dim`-th dimension.
"""
@inline topology(grid, dim) = topology(grid)[dim]

"""
    architecture(grid::AbstractGrid)

Return the architecture (CPU or GPU) that the `grid` lives on.
"""
@inline architecture(grid::AbstractGrid) = grid.architecture

"""
    size(grid)

Return a 3-tuple of the number of "center" cells on a grid in (x, y, z).
Center cells have the location (Center, Center, Center).
"""
@inline Base.size(grid::AbstractGrid) = (grid.Nx, grid.Ny, grid.Nz)
Base.eltype(::AbstractGrid{FT}) where FT = FT
Base.eltype(::Type{<:Oceananigans.Grids.AbstractGrid{FT}}) where FT = FT
Base.eps(::AbstractGrid{FT}) where FT = eps(FT)

function Base.:(==)(grid1::AbstractGrid, grid2::AbstractGrid)
    #check if grids are of the same type
    !isa(grid2, typeof(grid1).name.wrapper) && return false

    topology(grid1) !== topology(grid2) && return false

    x1, y1, z1 = nodes(grid1, (Face(), Face(), Face()))
    x2, y2, z2 = nodes(grid2, (Face(), Face(), Face()))

    CUDA.@allowscalar return x1 == x2 && y1 == y2 && z1 == z2
end

"""
    halo_size(grid)

Return a 3-tuple with the number of halo cells on either side of the
domain in (x, y, z).
"""
halo_size(grid) = (grid.Hx, grid.Hy, grid.Hz)
halo_size(grid, d) = halo_size(grid)[d]

@inline Base.size(grid::AbstractGrid, d::Int) = size(grid)[d]

grid_name(grid::AbstractGrid) = typeof(grid).name.wrapper

