using Oceananigans.Operators: Δrᵃᵃᶠ, Δrᵃᵃᶜ, Δzᵃᵃᶠ, Δzᵃᵃᶜ, Δr⁻¹ᵃᵃᶠ, Δr⁻¹ᵃᵃᶜ, Δz⁻¹ᵃᵃᶠ, Δz⁻¹ᵃᵃᶜ
using Oceananigans.Operators: Δxᶠᵃᵃ, Δxᶜᵃᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ, Δxᶜᶜᵃ
using Oceananigans.Operators: Δx⁻¹ᶠᵃᵃ, Δx⁻¹ᶜᵃᵃ, Δy⁻¹ᵃᶠᵃ, Δy⁻¹ᵃᶜᵃ
using Oceananigans.Operators: Δyᵃᶠᵃ, Δyᵃᶜᵃ, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶠᵃ, Δyᶜᶜᵃ
using Oceananigans.Operators: Δλᶜᵃᵃ, Δλᶜᶜᵃ, Δλᶜᶠᵃ, Δλᶠᵃᵃ, Δλᶠᶜᵃ, Δλᶠᶠᵃ, Δφᵃᶜᵃ, Δφᵃᶠᵃ, Δφᵃᶠᵃ, Δφᵃᶠᵃ, Δφᶜᶜᵃ, Δφᶜᶠᵃ, Δφᶠᶜᵃ, Δφᶠᶠᵃ
using Oceananigans.Operators: Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ, Azᶜᶜᵃ

using Oceananigans.Operators: Operators, intrinsic_vector, extrinsic_vector, rotation_angle

# Grid metrics for ImmersedBoundaryGrid
#
# The grid metric functions "specialized" for the underlying grids in the
# Operators module are extended for immersed boundary grids here.
#
# The other grid metric functions (for example 3D metrics) are general and do not need
# extension for "full-cell" immersed boundaries.
#
# However, for non "full-cell" immersed boundaries, grid metric functions
# must be extended for the specific immersed boundary grid in question.

# Vertical spacings

@inline Operators.Δrᵃᵃᶠ(i, j, k, ibg::IBG) = Δrᵃᵃᶠ(i, j, k, ibg.underlying_grid)
@inline Operators.Δrᵃᵃᶜ(i, j, k, ibg::IBG) = Δrᵃᵃᶜ(i, j, k, ibg.underlying_grid)
@inline Operators.Δzᵃᵃᶜ(i, j, k, ibg::IBG) = Δzᵃᵃᶜ(i, j, k, ibg.underlying_grid)
@inline Operators.Δzᵃᵃᶠ(i, j, k, ibg::IBG) = Δzᵃᵃᶠ(i, j, k, ibg.underlying_grid)
@inline Operators.Δzᵃᵃᶜ(i::AbstractArray, j::AbstractArray, k::AbstractArray, ibg::IBG) = Δzᵃᵃᶜ(i, j, k, ibg.underlying_grid)
@inline Operators.Δzᵃᵃᶠ(i::AbstractArray, j::AbstractArray, k::AbstractArray, ibg::IBG) = Δzᵃᵃᶠ(i, j, k, ibg.underlying_grid)

@inline Operators.Δr⁻¹ᵃᵃᶠ(i, j, k, ibg::IBG) = Δr⁻¹ᵃᵃᶠ(i, j, k, ibg.underlying_grid)
@inline Operators.Δr⁻¹ᵃᵃᶜ(i, j, k, ibg::IBG) = Δr⁻¹ᵃᵃᶜ(i, j, k, ibg.underlying_grid)
@inline Operators.Δz⁻¹ᵃᵃᶜ(i, j, k, ibg::IBG) = Δz⁻¹ᵃᵃᶜ(i, j, k, ibg.underlying_grid)
@inline Operators.Δz⁻¹ᵃᵃᶠ(i, j, k, ibg::IBG) = Δz⁻¹ᵃᵃᶠ(i, j, k, ibg.underlying_grid)

# 1D Horizontal spacings

@inline Operators.Δxᶠᵃᵃ(i, j, k, ibg::RGIBG) = Δxᶠᵃᵃ(i, j, k, ibg.underlying_grid)
@inline Operators.Δxᶜᵃᵃ(i, j, k, ibg::RGIBG) = Δxᶜᵃᵃ(i, j, k, ibg.underlying_grid)

@inline Operators.Δyᵃᶠᵃ(i, j, k, ibg::Union{RGIBG, LLIBG}) = Δyᵃᶠᵃ(i, j, k, ibg.underlying_grid)
@inline Operators.Δyᵃᶜᵃ(i, j, k, ibg::Union{RGIBG, LLIBG}) = Δyᵃᶜᵃ(i, j, k, ibg.underlying_grid)

@inline Operators.Δλᶜᵃᵃ(i, j, k, ibg::LLIBG) = Δλᶜᵃᵃ(i, j, k, ibg.underlying_grid)
@inline Operators.Δλᶠᵃᵃ(i, j, k, ibg::LLIBG) = Δλᶠᵃᵃ(i, j, k, ibg.underlying_grid)
@inline Operators.Δφᵃᶜᵃ(i, j, k, ibg::LLIBG) = Δφᵃᶜᵃ(i, j, k, ibg.underlying_grid)
@inline Operators.Δφᵃᶠᵃ(i, j, k, ibg::LLIBG) = Δφᵃᶠᵃ(i, j, k, ibg.underlying_grid)

# 2D Horizontal spacings

@inline Operators.Δxᶜᶠᵃ(i, j, k, ibg::Union{LLIBG, OSIBG}) = Δxᶜᶠᵃ(i, j, k, ibg.underlying_grid)
@inline Operators.Δxᶠᶜᵃ(i, j, k, ibg::Union{LLIBG, OSIBG}) = Δxᶠᶜᵃ(i, j, k, ibg.underlying_grid)
@inline Operators.Δxᶠᶠᵃ(i, j, k, ibg::Union{LLIBG, OSIBG}) = Δxᶠᶠᵃ(i, j, k, ibg.underlying_grid)
@inline Operators.Δxᶜᶜᵃ(i, j, k, ibg::Union{LLIBG, OSIBG}) = Δxᶜᶜᵃ(i, j, k, ibg.underlying_grid)

@inline Operators.Δyᶜᶜᵃ(i, j, k, ibg::OSIBG) = Δyᶜᶜᵃ(i, j, k, ibg.underlying_grid)
@inline Operators.Δyᶠᶜᵃ(i, j, k, ibg::OSIBG) = Δyᶠᶜᵃ(i, j, k, ibg.underlying_grid)
@inline Operators.Δyᶜᶠᵃ(i, j, k, ibg::OSIBG) = Δyᶜᶠᵃ(i, j, k, ibg.underlying_grid)
@inline Operators.Δyᶠᶠᵃ(i, j, k, ibg::OSIBG) = Δyᶠᶠᵃ(i, j, k, ibg.underlying_grid)

@inline Operators.Δλᶜᶜᵃ(i, j, k, ibg::OSIBG) = Δλᶜᶜᵃ(i, j, k, ibg.underlying_grid)
@inline Operators.Δλᶠᶜᵃ(i, j, k, ibg::OSIBG) = Δλᶠᶜᵃ(i, j, k, ibg.underlying_grid)
@inline Operators.Δλᶜᶠᵃ(i, j, k, ibg::OSIBG) = Δλᶜᶠᵃ(i, j, k, ibg.underlying_grid)
@inline Operators.Δλᶠᶠᵃ(i, j, k, ibg::OSIBG) = Δλᶠᶠᵃ(i, j, k, ibg.underlying_grid)

@inline Operators.Δφᶜᶜᵃ(i, j, k, ibg::OSIBG) = Δφᶜᶜᵃ(i, j, k, ibg.underlying_grid)
@inline Operators.Δφᶠᶜᵃ(i, j, k, ibg::OSIBG) = Δφᶠᶜᵃ(i, j, k, ibg.underlying_grid)
@inline Operators.Δφᶜᶠᵃ(i, j, k, ibg::OSIBG) = Δφᶜᶠᵃ(i, j, k, ibg.underlying_grid)
@inline Operators.Δφᶠᶠᵃ(i, j, k, ibg::OSIBG) = Δφᶠᶠᵃ(i, j, k, ibg.underlying_grid)

# Areas

@inline Operators.Azᶠᶜᵃ(i, j, k, ibg::Union{LLIBG, OSIBG}) = Azᶠᶜᵃ(i, j, k, ibg.underlying_grid)
@inline Operators.Azᶜᶠᵃ(i, j, k, ibg::Union{LLIBG, OSIBG}) = Azᶜᶠᵃ(i, j, k, ibg.underlying_grid)
@inline Operators.Azᶠᶠᵃ(i, j, k, ibg::Union{LLIBG, OSIBG}) = Azᶠᶠᵃ(i, j, k, ibg.underlying_grid)
@inline Operators.Azᶜᶜᵃ(i, j, k, ibg::Union{LLIBG, OSIBG}) = Azᶜᶜᵃ(i, j, k, ibg.underlying_grid)

# Reciprocal spacings and areas

@inline Operators.Δx⁻¹ᶠᵃᵃ(i, j, k, ibg::RGIBG) = Δx⁻¹ᶠᵃᵃ(i, j, k, ibg.underlying_grid)
@inline Operators.Δx⁻¹ᶜᵃᵃ(i, j, k, ibg::RGIBG) = Δx⁻¹ᶜᵃᵃ(i, j, k, ibg.underlying_grid)

@inline Operators.Δy⁻¹ᵃᶠᵃ(i, j, k, ibg::Union{RGIBG, LLIBG}) = Δy⁻¹ᵃᶠᵃ(i, j, k, ibg.underlying_grid)
@inline Operators.Δy⁻¹ᵃᶜᵃ(i, j, k, ibg::Union{RGIBG, LLIBG}) = Δy⁻¹ᵃᶜᵃ(i, j, k, ibg.underlying_grid)

for LX in (:ᶜ, :ᶠ), LY in (:ᶜ, :ᶠ)
    Δx⁻¹ˡˡᵃ = Symbol(:Δx⁻¹, LX, LY, :ᵃ)
    Δy⁻¹ˡˡᵃ = Symbol(:Δy⁻¹, LX, LY, :ᵃ)
    Az⁻¹ˡˡᵃ = Symbol(:Az⁻¹, LX, LY, :ᵃ)

    @eval begin
        @inline Operators.$Δx⁻¹ˡˡᵃ(i, j, k, ibg::Union{LLIBG, OSIBG}) = Operators.$Δx⁻¹ˡˡᵃ(i, j, k, ibg.underlying_grid)
        @inline Operators.$Δy⁻¹ˡˡᵃ(i, j, k, ibg::OSIBG)               = Operators.$Δy⁻¹ˡˡᵃ(i, j, k, ibg.underlying_grid)
        @inline Operators.$Az⁻¹ˡˡᵃ(i, j, k, ibg::Union{LLIBG, OSIBG}) = Operators.$Az⁻¹ˡˡᵃ(i, j, k, ibg.underlying_grid)
    end
end

# Extend both 2D and 3D methods
@inline Operators.intrinsic_vector(i, j, k, ibg::IBG, u, v) = intrinsic_vector(i, j, k, ibg.underlying_grid, u, v)
@inline Operators.extrinsic_vector(i, j, k, ibg::IBG, u, v) = extrinsic_vector(i, j, k, ibg.underlying_grid, u, v)

@inline Operators.intrinsic_vector(i, j, k, ibg::IBG, u, v, w) = intrinsic_vector(i, j, k, ibg.underlying_grid, u, v, w)
@inline Operators.extrinsic_vector(i, j, k, ibg::IBG, u, v, w) = extrinsic_vector(i, j, k, ibg.underlying_grid, u, v, w)
@inline Operators.rotation_angle(i, j, ibg::IBG, ℓx, ℓy) = rotation_angle(i, j, ibg.underlying_grid, ℓx, ℓy)

# Coordinate-line angles and orthogonality
for LX in (:ᶜ, :ᶠ), LY in (:ᶜ, :ᶠ), trig in (:cosα, :sinα)
    angle = Symbol(trig, LX, LY, :ᵃ)
    @eval @inline Operators.$angle(i, j, k, ibg::IBG) = Operators.$angle(i, j, k, ibg.underlying_grid)
end

Grids.isorthogonal(ibg::IBG) = Grids.isorthogonal(ibg.underlying_grid)
