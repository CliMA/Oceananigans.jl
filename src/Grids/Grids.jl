module Grids

export
    AbstractTopology, Periodic, Bounded, Flat, topology,
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
    Flat

Grid topology for flat dimensions, generally with one grid point, along which the solution
is uniform and does not vary.
"""
struct Flat <: AbstractTopology end

"""
    AbstractGrid{FT, TX, TY, TZ}

Abstract supertype for grids with elements of type `FT` and topology `{TX, TY, TZ}`.
"""
abstract type AbstractGrid{FT, TX, TY, TZ} end

eltype(::AbstractGrid{FT}) where FT = FT
topology(::AbstractGrid{FT, TX, TY, TZ}) where {FT, TX, TY, TZ} = (TX(), TY(), TZ())
topology(grid, dim) = topology(grid)[dim]

size(grid::AbstractGrid) = (grid.Nx, grid.Ny, grid.Nz)
length(grid::AbstractGrid) = (grid.Lx, grid.Ly, grid.Lz)
halo_size(grid) = (grid.Hx, grid.Hy, grid.Hz)

total_size(a) = size(a) # fallback

total_size(grid::AbstractGrid) = (grid.Nx + grid.Hx, 
                                  grid.Ny + grid.Hy, 
                                  grid.Nz + grid.Hz)

include("grid_utils.jl")
include("regular_cartesian_grid.jl")
include("vertically_stretched_cartesian_grid.jl")

end
