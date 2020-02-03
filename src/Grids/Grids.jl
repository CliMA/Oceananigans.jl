module Grids

export AbstractGrid, RegularCartesianGrid, VerticallyStretchedCartesianGrid

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
abstract type AbstractGrid{FT} end

include("grid_utils.jl")
include("regular_cartesian_grid.jl")
include("vertically_stretched_cartesian_grid.jl")

end
