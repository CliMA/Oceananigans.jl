using Oceananigans.AbstractOperations: GridMetricOperation

import Oceananigans.Operators: Δrᵃᵃᶠ, Δrᵃᵃᶜ, Δzᵃᵃᶠ, Δzᵃᵃᶜ
import Oceananigans.Operators: Δxᶠᵃᵃ, Δxᶜᵃᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ, Δxᶜᶜᵃ
import Oceananigans.Operators: Δyᵃᶠᵃ, Δyᵃᶜᵃ, Δyᶠᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶠᵃ, Δyᶜᶜᵃ
import Oceananigans.Operators: Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ, Azᶜᶜᵃ

import Oceananigans.Operators: intrinsic_vector, extrinsic_vector

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

@inline Δrᵃᵃᶠ(i, j, k, ibg::IBG) = Δrᵃᵃᶠ(i, j, k, ibg.underlying_grid)
@inline Δrᵃᵃᶜ(i, j, k, ibg::IBG) = Δrᵃᵃᶜ(i, j, k, ibg.underlying_grid)
@inline Δzᵃᵃᶜ(i, j, k, ibg::IBG) = Δzᵃᵃᶜ(i, j, k, ibg.underlying_grid)
@inline Δzᵃᵃᶠ(i, j, k, ibg::IBG) = Δzᵃᵃᶠ(i, j, k, ibg.underlying_grid)

# 1D Horizontal spacings

@inline Δxᶠᵃᵃ(i, j, k, ibg::RGIBG) = Δxᶠᵃᵃ(i, j, k, ibg.underlying_grid)
@inline Δxᶜᵃᵃ(i, j, k, ibg::RGIBG) = Δxᶜᵃᵃ(i, j, k, ibg.underlying_grid)

@inline Δyᵃᶠᵃ(i, j, k, ibg::Union{RGIBG, LLIBG}) = Δyᵃᶠᵃ(i, j, k, ibg.underlying_grid)
@inline Δyᵃᶜᵃ(i, j, k, ibg::Union{RGIBG, LLIBG}) = Δyᵃᶜᵃ(i, j, k, ibg.underlying_grid)

@inline Δλᶜᵃᵃ(i, j, k, ibg::LLIBG) = Δλᶜᵃᵃ(i, j, k, ibg.underlying_grid)
@inline Δλᶠᵃᵃ(i, j, k, ibg::LLIBG) = Δλᶠᵃᵃ(i, j, k, ibg.underlying_grid)
@inline Δφᵃᶜᵃ(i, j, k, ibg::LLIBG) = Δφᵃᶜᵃ(i, j, k, ibg.underlying_grid)
@inline Δφᵃᶠᵃ(i, j, k, ibg::LLIBG) = Δφᵃᶠᵃ(i, j, k, ibg.underlying_grid)

# 2D Horizontal spacings

@inline Δxᶜᶠᵃ(i, j, k, ibg::Union{LLIBG, OSIBG}) = Δxᶜᶠᵃ(i, j, k, ibg.underlying_grid)
@inline Δxᶠᶜᵃ(i, j, k, ibg::Union{LLIBG, OSIBG}) = Δxᶠᶜᵃ(i, j, k, ibg.underlying_grid)
@inline Δxᶠᶠᵃ(i, j, k, ibg::Union{LLIBG, OSIBG}) = Δxᶠᶠᵃ(i, j, k, ibg.underlying_grid)
@inline Δxᶜᶜᵃ(i, j, k, ibg::Union{LLIBG, OSIBG}) = Δxᶜᶜᵃ(i, j, k, ibg.underlying_grid)

@inline Δyᶜᶜᵃ(i, j, k, ibg::OSIBG) = Δyᶜᶜᵃ(i, j, k, ibg.underlying_grid)
@inline Δyᶠᶜᵃ(i, j, k, ibg::OSIBG) = Δyᶠᶜᵃ(i, j, k, ibg.underlying_grid)
@inline Δyᶜᶠᵃ(i, j, k, ibg::OSIBG) = Δyᶜᶠᵃ(i, j, k, ibg.underlying_grid)
@inline Δyᶠᶠᵃ(i, j, k, ibg::OSIBG) = Δyᶠᶠᵃ(i, j, k, ibg.underlying_grid)

@inline Δλᶜᶜᵃ(i, j, k, ibg::OSIBG) = Δλᶜᶜᵃ(i, j, k, ibg.underlying_grid)
@inline Δλᶠᶜᵃ(i, j, k, ibg::OSIBG) = Δλᶠᶜᵃ(i, j, k, ibg.underlying_grid)
@inline Δλᶜᶠᵃ(i, j, k, ibg::OSIBG) = Δλᶜᶠᵃ(i, j, k, ibg.underlying_grid)
@inline Δλᶠᶠᵃ(i, j, k, ibg::OSIBG) = Δλᶠᶠᵃ(i, j, k, ibg.underlying_grid)

@inline Δφᶜᶜᵃ(i, j, k, ibg::OSIBG) = Δφᶜᶜᵃ(i, j, k, ibg.underlying_grid)
@inline Δφᶠᶜᵃ(i, j, k, ibg::OSIBG) = Δφᶠᶜᵃ(i, j, k, ibg.underlying_grid)
@inline Δφᶜᶠᵃ(i, j, k, ibg::OSIBG) = Δφᶜᶠᵃ(i, j, k, ibg.underlying_grid)
@inline Δφᶠᶠᵃ(i, j, k, ibg::OSIBG) = Δφᶠᶠᵃ(i, j, k, ibg.underlying_grid)

# Areas

@inline Azᶠᶜᵃ(i, j, k, ibg::Union{LLIBG, OSIBG}) = Azᶠᶜᵃ(i, j, k, ibg.underlying_grid)
@inline Azᶜᶠᵃ(i, j, k, ibg::Union{LLIBG, OSIBG}) = Azᶜᶠᵃ(i, j, k, ibg.underlying_grid)
@inline Azᶠᶠᵃ(i, j, k, ibg::Union{LLIBG, OSIBG}) = Azᶠᶠᵃ(i, j, k, ibg.underlying_grid)
@inline Azᶜᶜᵃ(i, j, k, ibg::Union{LLIBG, OSIBG}) = Azᶜᶜᵃ(i, j, k, ibg.underlying_grid)

# Extend both 2D and 3D methods
@inline intrinsic_vector(i, j, k, ibg::IBG, u, v) = intrinsic_vector(i, j, k, ibg.underlying_grid, u, v)
@inline extrinsic_vector(i, j, k, ibg::IBG, u, v) = extrinsic_vector(i, j, k, ibg.underlying_grid, u, v)

@inline intrinsic_vector(i, j, k, ibg::IBG, u, v, w) = intrinsic_vector(i, j, k, ibg.underlying_grid, u, v, w)
@inline extrinsic_vector(i, j, k, ibg::IBG, u, v, w) = extrinsic_vector(i, j, k, ibg.underlying_grid, u, v, w)
