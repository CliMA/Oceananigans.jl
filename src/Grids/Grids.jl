module Grids

export Center, Face
export AbstractTopology, topology
export Periodic, Bounded, Flat, FullyConnected, LeftConnected, RightConnected
export RightFaceFolded, RightCenterFolded
export LeftConnectedRightCenterFolded, LeftConnectedRightFaceFolded
export LeftConnectedRightCenterConnected, LeftConnectedRightFaceConnected
export AbstractGrid, AbstractUnderlyingGrid, halo_size, total_size
export RectilinearGrid
export AbstractCurvilinearGrid, AbstractHorizontallyCurvilinearGrid
export XFlatGrid, YFlatGrid, ZFlatGrid
export XRegularRG, YRegularRG, ZRegularRG, XYRegularRG, XYZRegularRG
export LatitudeLongitudeGrid, XRegularLLG, YRegularLLG, ZRegularLLG
export OrthogonalSphericalShellGrid, ZRegOrthogonalSphericalShellGrid
export MutableVerticalDiscretization
export ExponentialDiscretization, ReferenceToStretchedDiscretization, PowerLawStretching, LinearStretching
export node, nodes
export ξnode, ηnode, rnode
export xnode, ynode, znode, λnode, φnode
export xnodes, ynodes, znodes, λnodes, φnodes, rnodes
export xspacings, yspacings, zspacings, λspacings, φspacings, rspacings
export minimum_xspacing, minimum_yspacing, minimum_zspacing
export static_column_depthᶜᶜᵃ, static_column_depthᶠᶜᵃ, static_column_depthᶜᶠᵃ, static_column_depthᶠᶠᵃ
export column_depthᶜᶜᵃ, column_depthᶠᶜᵃ, column_depthᶜᶠᵃ, column_depthᶠᶠᵃ
export offset_data, new_data
export on_architecture

using Adapt: Adapt, adapt
using GPUArraysCore: @allowscalar
using OffsetArrays: OffsetArray
using Printf: @sprintf
using DocStringExtensions: FIELDS
using BFloat16s: BFloat16

using Oceananigans: Oceananigans
using Oceananigans.Utils: Utils
using Oceananigans.Architectures: Architectures, AbstractSerialArchitecture, architecture, on_architecture

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
    FullyConnected

Grid topology for dimensions that are connected to other models or domains.
"""
struct FullyConnected <: AbstractTopology end

"""
    LeftConnected

Grid topology for dimensions that are connected to other models or domains only on the left (the other direction is bounded)
"""
struct LeftConnected <: AbstractTopology end

"""
    RightConnected

Grid topology for dimensions that are connected to other models or domains only on the right (the other direction is bounded)
"""
struct RightConnected <: AbstractTopology end

"""
    RightFaceFolded

Grid topology for tripolar F-point pivot connection
(folded north boundary along face locations).
"""
struct RightFaceFolded <: AbstractTopology end

"""
    RightCenterFolded

Grid topology for tripolar U-point pivot connection.
(folded north boundary along center locations).
"""
struct RightCenterFolded <: AbstractTopology end

"""
    LeftConnectedRightCenterFolded

Local grid topology for the northernmost y-rank of a 1×N distributed tripolar grid
with U-point pivot (serial fold). Connected to the south neighbor on the left,
center-folded on the right (north).
"""
struct LeftConnectedRightCenterFolded <: AbstractTopology end

"""
    LeftConnectedRightFaceFolded

Local grid y topology for the northernmost y-rank of a 1×N distributed tripolar grid
with F-point pivot (serial fold). Connected to the south neighbor on the left,
face-folded on the right (north). Face-extended (Ny+1 Face points in y).
"""
struct LeftConnectedRightFaceFolded <: AbstractTopology end

"""
    LeftConnectedRightCenterConnected

Local grid y topology for the northernmost y-rank of an M×N distributed tripolar grid
with U-point pivot (distributed zipper).
"""
struct LeftConnectedRightCenterConnected <: AbstractTopology end

"""
    LeftConnectedRightFaceConnected

Local grid y topology for the northernmost y-rank of an M×N distributed tripolar grid
with F-point pivot (distributed zipper).
Face-extended (Ny+1 Face points in y).
"""
struct LeftConnectedRightFaceConnected <: AbstractTopology end

#####
##### Directions (for tilted domains)
#####

abstract type AbstractDirection end

struct XDirection <: AbstractDirection end
struct YDirection <: AbstractDirection end
struct ZDirection <: AbstractDirection end

struct NegativeZDirection <: AbstractDirection end

const F = Face
const C = Center

include("abstract_grid.jl")
include("vertical_discretization.jl")
include("grid_utils.jl")
include("coordinate_utils.jl")
include("nodes_and_spacings.jl")
include("zeros_and_ones.jl")
include("new_data.jl")
include("inactive_node.jl")
include("automatic_halo_sizing.jl")
include("input_validation.jl")
include("grid_generation.jl")
include("rectilinear_grid.jl")
include("orthogonal_spherical_shell_grid.jl")
include("latitude_longitude_grid.jl")
include("coordinate_transformations.jl")

end # module
