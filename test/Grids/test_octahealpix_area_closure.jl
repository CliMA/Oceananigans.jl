using Test
using Oceananigans
using Oceananigans.Operators: Azᶜᶜᶜ, Vᶜᶜᶜ

# Verify global geometric closure on OctaHEALPix grids (§3.5):
# - Sum of cell areas matches 4π R² (whole sphere).
# - Cell volumes integrate to the shell volume 4π R² Δr.
# This protects against degenerate or duplicated/lost cells in the grid
# construction.

function octahealpix_area_sum(grid)
    total_area = zero(eltype(grid))
    for j in 1:size(grid, 2), i in 1:size(grid, 1)
        total_area += Azᶜᶜᶜ(i, j, 1, grid)
    end
    return total_area
end

function octahealpix_volume_sum(grid)
    total_volume = zero(eltype(grid))
    for j in 1:size(grid, 2), i in 1:size(grid, 1)
        total_volume += Vᶜᶜᶜ(i, j, 1, grid)
    end
    return total_volume
end

@testset "OctaHEALPix area and volume closure" begin
    for FT in (Float32, Float64)
        for N in (4, 8, 16)
            radius = one(FT)
            thickness = convert(FT, 1//3)
            grid = SphericalShellGrid(CPU(), FT;
                                      mapping = OctaHEALPixMapping(N),
                                      z = (zero(FT), thickness),
                                      radius = radius,
                                      halo = (5, 5, 3))

            expected_area = 4 * convert(FT, π) * radius^2
            expected_volume = expected_area * thickness

            area = octahealpix_area_sum(grid)
            volume = octahealpix_volume_sum(grid)

            # OctaHEALPix is equal-area by construction so the sum should
            # equal 4π R² to roundoff regardless of N. Use FT precision.
            tolerance = 100eps(FT) * expected_area

            @test isapprox(area, expected_area; atol = tolerance)
            @test isapprox(volume, expected_volume; atol = 100eps(FT) * expected_volume)
        end
    end
end
