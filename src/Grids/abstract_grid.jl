"""
    AbstractGrid{FT, TX, TY, TZ, Arch, SZ}

Abstract supertype for grids with elements of type `FT`, topology `{TX, TY, TZ}`,
`Arch`itecture, and static size descriptor `SZ`. `SZ` is either `Nothing` (the interior
and halo sizes are read from the `grid`'s fields at runtime) or a [`GridSize`](@ref),
"""
abstract type AbstractGrid{FT, TX, TY, TZ, Arch, SZ} end

grid(g::AbstractGrid) = g

"""
    AbstractUnderlyingGrid{FT, TX, TY, TZ, CZ, Arch, SZ}

Abstract supertype for "primary" grids (as opposed to grids with immersed boundaries)
with elements of type `FT`, topology `{TX, TY, TZ}`, vertical coordinate `CZ`, `Arch`itecture,
and static size descriptor `SZ`.
"""
abstract type AbstractUnderlyingGrid{FT, TX, TY, TZ, CZ, Arch, SZ} <: AbstractGrid{FT, TX, TY, TZ, Arch, SZ} end

"""
    AbstractCurvilinearGrid{FT, TX, TY, TZ, CZ, Arch, SZ}

Abstract supertype for curvilinear grids with elements of type `FT`,
topology `{TX, TY, TZ}`, vertical coordinate `CZ`, `Arch`itecture, and static size descriptor `SZ`.
"""
abstract type AbstractCurvilinearGrid{FT, TX, TY, TZ, CZ, Arch, SZ} <: AbstractUnderlyingGrid{FT, TX, TY, TZ, CZ, Arch, SZ} end

"""
    AbstractHorizontallyCurvilinearGrid{FT, TX, TY, TZ, CZ, Arch, SZ}

Abstract supertype for horizontally-curvilinear grids with elements of type `FT`,
topology `{TX, TY, TZ}`, vertical coordinate `CZ`, `Arch`itecture, and static size descriptor `SZ`.
"""
abstract type AbstractHorizontallyCurvilinearGrid{FT, TX, TY, TZ, CZ, Arch, SZ} <: AbstractCurvilinearGrid{FT, TX, TY, TZ, CZ, Arch, SZ} end

const XFlatGrid = AbstractGrid{<:Any, Flat}
const YFlatGrid = AbstractGrid{<:Any, <:Any, Flat}
const ZFlatGrid = AbstractGrid{<:Any, <:Any, <:Any, Flat}

const XYFlatGrid = AbstractGrid{<:Any, Flat, Flat}
const XZFlatGrid = AbstractGrid{<:Any, Flat, <:Any, Flat}
const YZFlatGrid = AbstractGrid{<:Any, <:Any, Flat, Flat}

const XYZFlatGrid = AbstractGrid{<:Any, Flat, Flat, Flat}

isrectilinear(grid) = false

# Fallback
@inline Utils.get_active_cells_map(::AbstractGrid, any_map_type) = nothing

"""
$(TYPEDSIGNATURES)

Return a tuple with the topology of the `grid` for each dimension.
"""
@inline topology(::AbstractGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} = (TX, TY, TZ)

"""
$(TYPEDSIGNATURES)

Return the topology of the `grid` for the `dim`-th dimension.
"""
@inline topology(grid, dim) = topology(grid)[dim]

"""
$(TYPEDSIGNATURES)

Return the architecture that the `grid` lives on.
"""
@inline Architectures.architecture(grid::AbstractGrid) = grid.architecture

"""
    GridSize{Nx, Ny, Nz, Hx, Hy, Hz}

Singleton type encoding a grid's interior size `(Nx, Ny, Nz)` and halo size `(Hx, Hy, Hz)` as type parameters.
Carrying it as the trailing grid type parameter makes both `size(grid)` and `halo_size(grid)` compile-time constants.
"""
struct GridSize{Nx, Ny, Nz, Hx, Hy, Hz}
    function GridSize(Nx, Ny, Nz, Hx, Hy, Hz)
        return new{Int(Nx), Int(Ny), Int(Nz), Int(Hx), Int(Hy), Int(Hz)}()
    end
end

"""
$(TYPEDSIGNATURES)

Return a 3-tuple of the number of "center" cells on a grid in (x, y, z).
Center cells have the location (Center, Center, Center).
"""
@inline Base.size(grid::AbstractGrid{<:Any, <:Any, <:Any, <:Any, <:Any, Nothing}) = map(Int, (grid.Nx, grid.Ny, grid.Nz))
@inline Base.size(grid::AbstractGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:GridSize{Nx, Ny, Nz}}) where {Nx, Ny, Nz} = (Nx, Ny, Nz)

Base.eltype(::AbstractGrid{FT}) where FT = FT
Base.eltype(::Type{<:AbstractGrid{FT}}) where FT = FT
Base.eps(::AbstractGrid{FT}) where FT = eps(FT)

function Base.:(==)(grid1::AbstractGrid, grid2::AbstractGrid)
    #check if grids are of the same type
    !isa(grid2, typeof(grid1).name.wrapper) && return false

    topology(grid1) !== topology(grid2) && return false

    x1, y1, z1 = nodes(grid1, (Face(), Face(), Face()))
    x2, y2, z2 = nodes(grid2, (Face(), Face(), Face()))

    @allowscalar return x1 == x2 && y1 == y2 && z1 == z2
end

"""
$(TYPEDSIGNATURES)

Return a 3-tuple with the number of halo cells on either side of the
domain in (x, y, z).
"""
@inline halo_size(grid::AbstractGrid{<:Any, <:Any, <:Any, <:Any, <:Any, Nothing}) = map(Int, (grid.Hx, grid.Hy, grid.Hz))
@inline halo_size(grid::AbstractGrid{<:Any, <:Any, <:Any, <:Any, <:Any, <:GridSize{Nx, Ny, Nz, Hx, Hy, Hz}}) where {Nx, Ny, Nz, Hx, Hy, Hz} = (Hx, Hy, Hz)

halo_size(grid, d) = halo_size(grid)[d]

@inline Base.size(grid::AbstractGrid, d::Int) = size(grid)[d]

grid_name(grid::AbstractGrid) = typeof(grid).name.wrapper
