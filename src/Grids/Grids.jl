module Grids

export Center, Face
export AbstractTopology, topology
export Periodic, Flat, Sides, Bounded, Closed, Open, Connected, CenterFolded, FaceFolded, CenterFoldConnected, FaceFoldConnected
export SerialFoldedTopology, SlabFoldedTopology, PencilFoldedTopology, DistributedFoldedTopology, FoldedTopology
export AbstractGrid, AbstractUnderlyingGrid, grid, halo_size, total_size
export slice
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

using Adapt: Adapt
using DocStringExtensions: FIELDS, TYPEDSIGNATURES
using GPUArraysCore: @allowscalar
using OffsetArrays: OffsetArray
using Printf: @sprintf
using BFloat16s: BFloat16

using Oceananigans: Oceananigans
using Oceananigans.Architectures: Architectures, AbstractSerialArchitecture, CPU,
                                  architecture, on_architecture
using Oceananigans.Utils: Utils, KernelParameters, worksize, unparameterized_type

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
    Flat

Grid topology for flat dimensions, generally with one grid point, along which the solution
is uniform and does not vary.
"""
struct Flat <: AbstractTopology end

"""
    Sides{L, R}

Grid topology for dimensions whose left side has topology `L` and right side has topology `R`.
The topology of a side is `Closed`, `Open`, `Connected`, `CenterFolded`, `FaceFolded`, `CenterFoldConnected`, or
`FaceFoldConnected`. A side topology used on its own, for example `Closed`, applies to both sides.
"""
struct Sides{L, R} <: AbstractTopology end

"""
    Closed

Topology of a side closed by a wall.
"""
struct Closed end

"""
    Open

Topology of a side where the domain continues into an exterior.
"""
struct Open end

"""
    Bounded

Grid topology for dimensions closed by walls on both sides.
"""
const Bounded = Sides{Closed, Closed}

"""
    Connected

Topology of a side connected to another rank or region.
"""
struct Connected end

"""
    CenterFolded

Topology of a side folded onto itself along cell centers (tripolar U-point and T-point pivots).
"""
struct CenterFolded end

"""
    FaceFolded

Topology of a side folded onto itself along cell faces (tripolar F-point pivot). The fold face belongs to the domain.
"""
struct FaceFolded end

"""
    CenterFoldConnected

Topology of a `CenterFolded` side whose fold partner lives on another rank.
"""
struct CenterFoldConnected end

"""
    FaceFoldConnected

Topology of a `FaceFolded` side whose fold partner lives on another rank.
"""
struct FaceFoldConnected end

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
