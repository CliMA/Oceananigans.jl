using Oceananigans.AbstractOperations: GridMetricOperation

import Oceananigans.Grids: coordinates
import Oceananigans.Operators: Δzᵃᵃᶠ, Δzᵃᵃᶜ, intrinsic_vector, extrinsic_vector

# Grid metrics for ImmersedBoundaryGrid
#
# All grid metrics are defined here.
#
# For non "full-cell" immersed boundaries, grid metric functions
# must be extended for the specific immersed boundary grid in question.

for LX in (:ᶜ, :ᶠ, :ᵃ), LY in (:ᶜ, :ᶠ, :ᵃ), LZ in (:ᶜ, :ᶠ, :ᵃ)
    for dir in (:x, :y, :z), operator in (:Δ, :A)

        metric = Symbol(operator, dir, LX, LY, LZ)
        @eval begin
            import Oceananigans.Operators: $metric
            @inline $metric(i, j, k, ibg::IBG) = $metric(i, j, k, ibg.underlying_grid)
        end
    end

    volume = Symbol(:V, LX, LY, LZ)
    @eval begin
        import Oceananigans.Operators: $volume
        @inline $volume(i, j, k, ibg::IBG) = $volume(i, j, k, ibg.underlying_grid)
    end
end

coordinates(grid::IBG) = coordinates(grid.underlying_grid)

@inline Δzᵃᵃᶠ(i, j, k, ibg::IBG) = Δzᵃᵃᶠ(i, j, k, ibg.underlying_grid)
@inline Δzᵃᵃᶜ(i, j, k, ibg::IBG) = Δzᵃᵃᶜ(i, j, k, ibg.underlying_grid)

# Extend both 2D and 3D methods
@inline intrinsic_vector(i, j, k, ibg::IBG, u, v) = intrinsic_vector(i, j, k, ibg.underlying_grid, u, v)
@inline extrinsic_vector(i, j, k, ibg::IBG, u, v) = extrinsic_vector(i, j, k, ibg.underlying_grid, u, v)

@inline intrinsic_vector(i, j, k, ibg::IBG, u, v, w) = intrinsic_vector(i, j, k, ibg.underlying_grid, u, v, w)
@inline extrinsic_vector(i, j, k, ibg::IBG, u, v, w) = extrinsic_vector(i, j, k, ibg.underlying_grid, u, v, w)