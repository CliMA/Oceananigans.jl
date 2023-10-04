module Coriolis

export
    FPlane, ConstantCartesianCoriolis, BetaPlane, NonTraditionalBetaPlane,
    HydrostaticSphericalCoriolis, ActiveCellEnstrophyConservingScheme,
    x_f_cross_U, y_f_cross_U, z_f_cross_U

using Printf
using Adapt
using Oceananigans.Grids
using Oceananigans.Operators

# Physical constants for constructors.
using Oceananigans.Grids: R_Earth
const Ω_Earth = 7.292115e-5 # [s⁻¹] https://en.wikipedia.org/wiki/Earth%27s_rotation#Angular_speed

"""
    AbstractRotation

Abstract supertype for parameters related to background rotation rates.
"""
abstract type AbstractRotation end

include("no_rotation.jl")
include("f_plane.jl")
include("constant_cartesian_coriolis.jl")
include("beta_plane.jl")
include("non_traditional_beta_plane.jl")
include("hydrostatic_spherical_coriolis.jl")

end # module
