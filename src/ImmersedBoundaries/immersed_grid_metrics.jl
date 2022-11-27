using Oceananigans.AbstractOperations: GridMetricOperation

import Oceananigans.Grids: return_metrics, min_Δx, min_Δy, min_Δz, xspacing, yspacing, zspacing

const c = Center()
const f = Face()
const IBG = ImmersedBoundaryGrid

# Grid metrics for ImmersedBoundaryGrid
#
# All grid metrics are defined here.
#
# For non "full-cell" immersed boundaries, grid metric functions
# must be extended for the specific immersed boundary grid in question.
#
for LX in (:ᶜ, :ᶠ), LY in (:ᶜ, :ᶠ), LZ in (:ᶜ, :ᶠ)
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

@inline Δzᵃᵃᶜ(i, j, k, ibg::IBG) = Δzᵃᵃᶜ(i, j, k, ibg.underlying_grid)
@inline Δzᵃᵃᶠ(i, j, k, ibg::IBG) = Δzᵃᵃᶠ(i, j, k, ibg.underlying_grid)


return_metrics(grid::IBG) = return_metrics(grid.underlying_grid)
xspacing(X, grid::IBG) = xspacing(X, grid.underlying_grid)
yspacing(Y, grid::IBG) = yspacing(Y, grid.underlying_grid)
zspacing(Z, grid::IBG) = zspacing(Z, grid.underlying_grid)
min_Δx(grid::IBG) = min_Δx(grid.underlying_grid)
min_Δy(grid::IBG) = min_Δy(grid.underlying_grid)
min_Δz(grid::IBG) = min_Δz(grid.underlying_grid)
