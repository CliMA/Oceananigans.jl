import Oceananigans.Grids: coordinates
import Oceananigans.Operators: intrinsic_vector, extrinsic_vector

# Grid metrics for ImmersedBoundaryGrid
#
# All grid metrics are defined here.
#
# For non "full-cell" immersed boundaries, grid metric functions
# must be extended for the specific immersed boundary grid in question.

# We need to extend only the metric explicitly defined in the `spacing_and_areas_and_volumes.jl` file. 
# These include the one-dimensional and two-dimensional (horizontal) spacings and the horizontal areas.

for L1 in (:ᶜ, :ᶠ)
    xspacing1D = Symbol(:Δx, L1, :ᵃ, :ᵃ)
    yspacing1D = Symbol(:Δy, :ᵃ, L1, :ᵃ)
    zspacing1D = Symbol(:Δz, :ᵃ, :ᵃ, L1)

    @eval begin
        import Oceananigans.Operators: $xspacing1D, $yspacing1D, $zspacing1D

        $xspacing1D(i, j, k, ibg::IBG) = $xspacing1D(i, j, k, ibg.underlying_grid)
        $yspacing1D(i, j, k, ibg::IBG) = $yspacing1D(i, j, k, ibg.underlying_grid)
        $zspacing1D(i, j, k, ibg::IBG) = $zspacing1D(i, j, k, ibg.underlying_grid)
    end

    for L2 in (:ᶜ, :ᶠ)
        xspacing2D = Symbol(:Δx, L1, L2, :ᵃ)
        yspacing2D = Symbol(:Δy, L2, L1, :ᵃ)
        zarea2D    = Symbol(:Az, L1, L2, :ᵃ)

        @eval begin
            import Oceananigans.Operators: $xspacing2D, $yspacing2D, $zarea2D
    
            $xspacing2D(i, j, k, ibg::IBG) = $xspacing2D(i, j, k, ibg.underlying_grid)
            $yspacing2D(i, j, k, ibg::IBG) = $yspacing2D(i, j, k, ibg.underlying_grid)
            $zarea2D(i, j, k, ibg::IBG) = $zarea2D(i, j, k, ibg.underlying_grid)
        end
    end
end

coordinates(grid::IBG) = coordinates(grid.underlying_grid)

# Extend both 2D and 3D methods
@inline intrinsic_vector(i, j, k, ibg::IBG, u, v) = intrinsic_vector(i, j, k, ibg.underlying_grid, u, v)
@inline extrinsic_vector(i, j, k, ibg::IBG, u, v) = extrinsic_vector(i, j, k, ibg.underlying_grid, u, v)

@inline intrinsic_vector(i, j, k, ibg::IBG, u, v, w) = intrinsic_vector(i, j, k, ibg.underlying_grid, u, v, w)
@inline extrinsic_vector(i, j, k, ibg::IBG, u, v, w) = extrinsic_vector(i, j, k, ibg.underlying_grid, u, v, w)