import Oceananigans.Grids: coordinates
import Oceananigans.Operators: intrinsic_vector, extrinsic_vector

# Grid metrics for ImmersedBoundaryGrid
#
# All grid metrics are defined here.
#
# For non "full-cell" immersed boundaries, grid metric functions
# must be extended for the specific immersed boundary grid in question.

# We need to extend only the metric explicitly defined in the `spacing_and_areas_and_volumes.jl` file. 
# These include all the spacings and the horizontal areas. 
# All the other metrics are calculated from these.

x_superscript(dir) = dir == :x ? (:ᶜ, :ᶠ) : (:ᶜ, :ᶠ, :ᵃ)
y_superscript(dir) = dir == :y ? (:ᶜ, :ᶠ) : (:ᶜ, :ᶠ, :ᵃ)
z_superscript(dir) = dir == :z ? (:ᶜ, :ᶠ) : (:ᶜ, :ᶠ, :ᵃ)

for dir in (:x, :y, :z)
    for LX in x_superscript(dir), LY in y_superscript(dir), LZ in z_superscript(dir)
        spacing = Symbol(:Δ, dir, LX, LY, LZ)
        @eval begin
            import Oceananigans.Operators: $spacing
            $spacing(i, j, k, ibg::IBG) = $spacing(i, j, k, ibg.underlying_grid)
        end
    end
end

for L1 in (:ᶜ, :ᶠ)
    for L2 in (:ᶜ, :ᶠ)
        zarea2D = Symbol(:Az, L1, L2, :ᵃ)
        @eval begin
            import Oceananigans.Operators: $zarea2D
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