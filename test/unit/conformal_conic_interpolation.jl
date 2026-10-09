include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.OrthogonalSphericalShellGrids: ConformalConicGrid, geographic_to_conformal_conic,
    conformal_conic_fractional_indices
using Oceananigans.Fields: interpolate!, interior
using Oceananigans.Grids: Center, Face, λnodes, φnodes

# Bilinear interpolation of a field that is linear in the projected (x, y)
# coordinates is exact, so the interpolant must reproduce a·x + b·y to round-off.
linear_in_projection(map, a, b) = (λ, φ, z) -> begin
    x, y = geographic_to_conformal_conic(map, λ, φ)
    a * x + b * y
end

@testset "ConformalConic source interpolation [$(typeof(arch)), $FT]" for arch in archs, FT in float_types
    source_grid = ConformalConicGrid(arch, FT;
                                     size = (40, 30, 2),
                                     center = (-95, 38),
                                     spacing = 20000,
                                     standard_parallels = (25, 25),
                                     z = (-1, 0),
                                     warn = false)

    map = source_grid.conformal_mapping

    # LatitudeLongitude target strictly inside the projected LCC rectangle.
    target_grid = LatitudeLongitudeGrid(arch, FT;
                                        size = (12, 10, 2),
                                        longitude = (-98, -92),
                                        latitude = (36, 40),
                                        z = (-1, 0))

    # Test 1 — constant field interpolates to the same constant (exact).
    source_constant = CenterField(source_grid)
    set!(source_constant, (λ, φ, z) -> convert(FT, 3.5))
    target_constant = CenterField(target_grid)
    interpolate!(target_constant, source_constant)
    @test maximum(abs.(Array(interior(target_constant)) .- FT(3.5))) < 10 * eps(FT)

    # Tests 2 & 3 — linear-in-projection field, at Center and at Face (u-like).
    a, b = convert(FT, 1e-6), convert(FT, -1e-6)
    f = linear_in_projection(map, a, b)
    tolerance = FT == Float64 ? 1e-9 : 1e-2

    for (TargetField, ℓx, ℓy) in ((CenterField, Center(), Center()),
                                  (XFaceField,  Face(),   Center()))
        source_field = TargetField(source_grid)
        set!(source_field, f)
        target_field = TargetField(target_grid)
        interpolate!(target_field, source_field)

        λt = Array(λnodes(target_grid, ℓx, ℓy, Center()))
        φt = Array(φnodes(target_grid, ℓx, ℓy, Center()))
        Nx, Ny = length(λt), length(φt)
        expected = [a * geographic_to_conformal_conic(map, λt[i], φt[j])[1] +
            b * geographic_to_conformal_conic(map, λt[i], φt[j])[2]
                    for i in 1:Nx, j in 1:Ny, k in 1:2]

        @test maximum(abs.(Array(interior(target_field)) .- expected)) < tolerance
    end

    # Type stability and (on CPU) allocation-freedom of the fractional-index helper.
    λ₀, φ₀ = FT(-95), FT(38)
    @test @inferred(conformal_conic_fractional_indices(λ₀, φ₀, source_grid, Center(), Center())) isa Tuple{FT, FT}

    if arch isa CPU
        conformal_conic_fractional_indices(λ₀, φ₀, source_grid, Center(), Center())
        @test (@allocated conformal_conic_fractional_indices(λ₀, φ₀, source_grid, Center(), Center())) == 0
    end
end
