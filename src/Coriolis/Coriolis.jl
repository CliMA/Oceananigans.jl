module Coriolis

export
    FPlane, ConstantCartesianCoriolis, BetaPlane, NonTraditionalBetaPlane,
    SphericalCoriolis, HydrostaticSphericalCoriolis,
    ActiveWeightedEnstrophyConserving, ActiveWeightedEnergyConserving, EENConserving,
    x_f_cross_U, y_f_cross_U, z_f_cross_U

using Printf: @sprintf
using Adapt: Adapt
using Oceananigans: Oceananigans
using Oceananigans.Grids: AbstractGrid, Center, Face, ynode, znode
using Oceananigans.Operators: active_weighted_ℑxyᶜᶠᶜ, active_weighted_ℑxyᶠᶜᶜ, ℑxᶜᵃᵃ, ℑxᶠᵃᵃ,
    ℑyᵃᶜᵃ, ℑyᵃᶠᵃ, ℑzᵃᵃᶜ, ℑzᵃᵃᶠ, ℑxyᶜᶠᵃ, ℑxyᶠᶜᵃ, ℑxzᶜᵃᶠ, Ay⁻¹ᶠᶜᶜ, Ax⁻¹ᶜᶠᶜ, Ax_qᶠᶜᶜ, Ay_qᶜᶠᶜ

using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid

"""
    AbstractRotation{S}

Abstract supertype for parameters related to background rotation rates.
`S` is the type of the scheme implemented.
"""
abstract type AbstractRotation{S} end

const face = Face()
const center = Center()

include("no_rotation.jl")
include("coriolis_schemes.jl")
include("f_plane.jl")
include("constant_cartesian_coriolis.jl")
include("beta_plane.jl")
include("non_traditional_beta_plane.jl")
include("spherical_coriolis.jl")

end # module
