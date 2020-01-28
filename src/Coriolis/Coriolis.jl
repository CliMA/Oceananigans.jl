module Coriolis

export
    FPlane, NonTraditionalFPlane, BetaPlane,
    x_f_cross_U, y_f_cross_U, z_f_cross_U

# Physical constants for FPlane and BetaPlane constructors.
const Ω_Earth = 7.292115e-5 # [s⁻¹] https://en.wikipedia.org/wiki/Earth%27s_rotation#Angular_speed
const R_Earth = 6371.0e3    # Mean radius of the Earth [m] https://en.wikipedia.org/wiki/Earth

"""
    AbstractRotation

Abstract supertype for parameters related to background rotation rates.
"""
abstract type AbstractRotation end

include("no_rotation.jl")
include("f_plane.jl")
include("beta_plane.jl")

end
