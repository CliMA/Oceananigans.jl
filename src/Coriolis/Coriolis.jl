module Coriolis

export
    FPlane, ConstantCartesianCoriolis, BetaPlane, NonTraditionalBetaPlane,
    HydrostaticSphericalCoriolis, ActiveCellEnstrophyConserving,
    x_f_cross_U, y_f_cross_U, z_f_cross_U

using Printf
using Adapt
using Oceananigans
using Oceananigans.Grids
using Oceananigans.Operators
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid

# Physical constants for constructors.
using Oceananigans.Grids: R_Earth

"Earth's rotation rate [s⁻¹]; see https://en.wikipedia.org/wiki/Earth%27s_rotation#Angular_speed"
const Ω_Earth = 7.292115e-5

"""
    AbstractRotation

Abstract supertype for parameters related to background rotation rates.
"""
abstract type AbstractRotation end

@inline active_weighted_ℑxyᶜᶠᵃ(i, j, k, grid, q, args...) = zero(grid)
@inline active_weighted_ℑxyᶠᶜᵃ(i, j, k, grid, q, args...) = zero(grid)

@inline not_peripheral_node(args...) = !peripheral_node(args...)

const face = Face()
const center = Center()

@inline function active_weighted_ℑxyᶜᶠᵃ(i, j, k, grid::ImmersedBoundaryGrid, q, args...)
    actives = ℑxyᶜᶠᵃ(i, j, k, grid, not_peripheral_node, face, center, center)
    mask = actives == 0
    return ifelse(mask, zero(grid), ℑxyᶜᶠᵃ(i, j, k, grid, q, args...) / actives)
end

@inline function active_weighted_ℑxyᶠᶜᵃ(i, j, k, grid::ImmersedBoundaryGrid, q, args...)
    actives = ℑxyᶜᶠᵃ(i, j, k, grid, not_peripheral_node, face, center, center)
    mask = actives == 0
    return ifelse(mask, zero(grid), ℑxyᶠᶜᵃ(i, j, k, grid, q, args...) / actives)
end


include("no_rotation.jl")
include("f_plane.jl")
include("constant_cartesian_coriolis.jl")
include("beta_plane.jl")
include("non_traditional_beta_plane.jl")
include("hydrostatic_spherical_coriolis.jl")

end # module
