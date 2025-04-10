module Grids

export Center, Face
export AbstractTopology, Periodic, Bounded, Flat, FullyConnected, LeftConnected, RightConnected, topology

export AbstractGrid, AbstractUnderlyingGrid, halo_size, total_size
export RectilinearGrid
export AbstractCurvilinearGrid, AbstractHorizontallyCurvilinearGrid
export XFlatGrid, YFlatGrid, ZFlatGrid
export XRegularRG, YRegularRG, ZRegularRG, XYRegularRG, XYZRegularRG
export LatitudeLongitudeGrid, XRegularLLG, YRegularLLG, ZRegularLLG
export OrthogonalSphericalShellGrid, ConformalCubedSphereGrid, ZRegOrthogonalSphericalShellGrid
export conformal_cubed_sphere_panel
export MutableVerticalDiscretization
export node, nodes
export ξnode, ηnode, rnode
export xnode, ynode, znode, λnode, φnode
export xnodes, ynodes, znodes, λnodes, φnodes, rnodes
export spacings
export xspacings, yspacings, zspacings, λspacings, φspacings, rspacings
export minimum_xspacing, minimum_yspacing, minimum_zspacing
export static_column_depthᶜᶜᵃ, static_column_depthᶠᶜᵃ, static_column_depthᶜᶠᵃ, static_column_depthᶠᶠᵃ
export column_depthᶜᶜᵃ, column_depthᶠᶜᵃ, column_depthᶜᶠᵃ, column_depthᶠᶠᵃ
export offset_data, new_data
export on_architecture

using CUDA
using CUDA: has_cuda
using Adapt
using OffsetArrays
using Printf

using Oceananigans
using Oceananigans.Architectures

import Base: size, length, eltype, show, -
import Oceananigans.Architectures: architecture, on_architecture

# Physical constants for constructors.
const R_Earth = 6371.0e3    # [m] Mean radius of the Earth https://en.wikipedia.org/wiki/Earth

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

#####
##### Directions (for tilted domains)
#####

abstract type AbstractDirection end

struct XDirection <: AbstractDirection end
struct YDirection <: AbstractDirection end
struct ZDirection <: AbstractDirection end

struct NegativeZDirection <: AbstractDirection end

include("abstract_grid.jl")
include("vertical_discretization.jl")
include("grid_utils.jl")
include("nodes_and_spacings.jl")
include("zeros_and_ones.jl")
include("new_data.jl")
include("inactive_node.jl")
include("automatic_halo_sizing.jl")
include("input_validation.jl")
include("grid_generation.jl")
include("rectilinear_grid.jl")
include("orthogonal_spherical_shell_grid.jl")
include("conformal_cubed_sphere_panel.jl")
include("latitude_longitude_grid.jl")

end # module
