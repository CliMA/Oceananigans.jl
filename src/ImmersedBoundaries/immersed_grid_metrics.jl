using Oceananigans.AbstractOperations: GridMetricOperation

import Oceananigans.Grids: return_metrics,
                           xspacing, yspacing, zspacing,
                           xspacings, yspacings, zspacings

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

xspacing(grid::IBG, args...) = xspacing(grid.underlying_grid, args...)
yspacing(grid::IBG, args...) = yspacing(grid.underlying_grid, args...)
zspacing(grid::IBG, args...) = zspacing(grid.underlying_grid, args...)

xspacings(grid::IBG, args...) = xspacings(grid.underlying_grid, args...)
yspacings(grid::IBG, args...) = yspacings(grid.underlying_grid, args...)
zspacings(grid::IBG, args...) = zspacings(grid.underlying_grid, args...)
