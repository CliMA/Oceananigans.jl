module Grids

export
    AbstractTopology, Periodic, Bounded, Singleton, topology,
    AbstractGrid, RegularCartesianGrid, VerticallyStretchedCartesianGrid

import Base: size, length, eltype, show

using Oceananigans

"""
    AbstractTopology

Abstract supertype for grid topologies.
"""
abstract type AbstractTopology end

"""
    Periodic

Grid topology for periodic dimensions.
"""
struct Periodic <: AbstractTopology end

"""
    Bounded

Grid topology for bounded dimensions. These could be wall-bounded dimensions
or dimensions
"""
struct Bounded <: AbstractTopology end

"""
    Singleton

Grid topology for singleton dimensions with one grid point.
"""
struct Singleton <: AbstractTopology end

"""
    AbstractGrid{FT, TX, TY, TZ}

Abstract supertype for grids with elements of type `FT` and topology `{TX, TY, TZ}`.
"""
abstract type AbstractGrid{FT, TX, TY, TZ} end

eltype(::AbstractGrid{FT}) where FT = FT
topology(::AbstractGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} = (TX, TY, TZ)

size(grid::AbstractGrid)   = (grid.Nx, grid.Ny, grid.Nz)
length(grid::AbstractGrid) = (grid.Lx, grid.Ly, grid.Lz)

include("grid_utils.jl")
include("regular_cartesian_grid.jl")
include("vertically_stretched_cartesian_grid.jl")

end
