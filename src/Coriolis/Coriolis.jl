module Coriolis

export
    FPlane, ConstantCartesianCoriolis, BetaPlane, NonTraditionalBetaPlane,
    SphericalCoriolis, HydrostaticSphericalCoriolis,
    ActiveCellEnstrophyConserving,
    x_f_cross_U, y_f_cross_U, z_f_cross_U

using Printf
using Adapt
using Oceananigans
using Oceananigans.Grids
using Oceananigans.Operators
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid

"""
    AbstractRotation

Abstract supertype for parameters related to background rotation rates.
"""
abstract type AbstractRotation end

const face = Face()
const center = Center()

include("no_rotation.jl")
include("f_plane.jl")
include("constant_cartesian_coriolis.jl")
include("beta_plane.jl")
include("non_traditional_beta_plane.jl")
include("spherical_coriolis.jl")

end # module
