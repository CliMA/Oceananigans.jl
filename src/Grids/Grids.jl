module Grids

export Center, Face
export AbstractTopology, Periodic, Bounded, Flat, Connected, topology

export AbstractGrid, AbstractUnderlyingGrid, halo_size, total_size
export AbstractRectilinearGrid, RectilinearGrid 
export XRegRectilinearGrid, YRegRectilinearGrid, ZRegRectilinearGrid, HRegRectilinearGrid, RegRectilinearGrid
export AbstractCurvilinearGrid, AbstractHorizontallyCurvilinearGrid
export LatitudeLongitudeGrid, XRegLatLonGrid, YRegLatLonGrid, ZRegLatLonGrid
export ConformalCubedSphereFaceGrid, ConformalCubedSphereGrid
export node, xnode, ynode, znode, xnodes, ynodes, znodes, nodes
export offset_data, new_data
export on_architecture

using CUDA
using Adapt
using OffsetArrays

using Oceananigans
using Oceananigans.Architectures

import Base: size, length, eltype, show
import Oceananigans.Architectures: architecture

#####
##### Abstract types
#####

"""
    Center

A type describing the location at the center of a grid cell.
"""
struct Center end

"""
	Face

A type describing the location at the face of a grid cell.
"""
struct Face end

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

Grid topology for bounded dimensions, e.g., wall-bounded dimensions.
"""
struct Bounded <: AbstractTopology end

"""
    Flat

Grid topology for flat dimensions, generally with one grid point, along which the solution
is uniform and does not vary.
"""
struct Flat <: AbstractTopology end

"""
    Connected

Grid topology for dimensions that are connected to other models or domains on both sides.
"""
const Connected = Periodic  # Right now we just need them to behave like Periodic dimensions except we change the boundary conditions.

"""
    AbstractGrid{FT, TX, TY, TZ}

Abstract supertype for grids with elements of type `FT` and topology `{TX, TY, TZ}`.
"""
abstract type AbstractGrid{FT, TX, TY, TZ, Arch} end

"""
    AbstractUnderlyingGrid{FT, TX, TY, TZ}

Abstract supertype for "primary" grids (as opposed to grids with immersed boundaries)
with elements of type `FT` and topology `{TX, TY, TZ}`.
"""
abstract type AbstractUnderlyingGrid{FT, TX, TY, TZ, Arch} <: AbstractGrid{FT, TX, TY, TZ, Arch} end

"""
    AbstractRectilinearGrid{FT, TX, TY, TZ}

Abstract supertype for rectilinear grids with elements of type `FT` and topology `{TX, TY, TZ}`.
"""
abstract type AbstractRectilinearGrid{FT, TX, TY, TZ, Arch} <: AbstractUnderlyingGrid{FT, TX, TY, TZ, Arch} end

"""
    AbstractCurvilinearGrid{FT, TX, TY, TZ}

Abstract supertype for curvilinear grids with elements of type `FT` and topology `{TX, TY, TZ}`.
"""
abstract type AbstractCurvilinearGrid{FT, TX, TY, TZ, Arch} <: AbstractUnderlyingGrid{FT, TX, TY, TZ, Arch} end

"""
    AbstractHorizontallyCurvilinearGrid{FT, TX, TY, TZ}

Abstract supertype for horizontally-curvilinear grids with elements of type `FT` and topology `{TX, TY, TZ}`.
"""
abstract type AbstractHorizontallyCurvilinearGrid{FT, TX, TY, TZ, Arch} <: AbstractCurvilinearGrid{FT, TX, TY, TZ, Arch} end

isrectilinear(grid) = false

include("grid_utils.jl")
include("zeros.jl")
include("new_data.jl")
include("grid_solid_nodes.jl")
include("automatic_halo_sizing.jl")
include("input_validation.jl")
include("grid_generation.jl")
include("rectilinear_grid.jl")
include("conformal_cubed_sphere_face_grid.jl")
include("latitude_longitude_grid.jl")

end
