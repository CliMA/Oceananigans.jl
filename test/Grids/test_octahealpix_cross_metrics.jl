using Test
using Oceananigans

function octahealpix_cross_metric_averaging_errors(FT, N)
    grid = SphericalShellGrid(CPU(), FT;
                              mapping = OctaHEALPixMapping(N),
                              z = (zero(FT), one(FT)),
                              radius = one(FT),
                              halo = (5, 5, 3))

    half = convert(FT, 1//2)

    g¹²ᶜᶜᵃ = grid.metrics.g¹²ᶜᶜᵃ
    G¹²ᶜᶜᵃ = grid.metrics.G¹²ᶜᶜᵃ
    g¹²ᶠᶜᵃ = grid.metrics.g¹²ᶠᶜᵃ
    G¹²ᶠᶜᵃ = grid.metrics.G¹²ᶠᶜᵃ
    g²¹ᶜᶠᵃ = grid.metrics.g²¹ᶜᶠᵃ
    G²¹ᶜᶠᵃ = grid.metrics.G²¹ᶜᶠᵃ

    max_g¹²ᶠᶜᵃ_error = zero(FT)
    max_G¹²ᶠᶜᵃ_error = zero(FT)
    max_g²¹ᶜᶠᵃ_error = zero(FT)
    max_G²¹ᶜᶠᵃ_error = zero(FT)
    x_count = 0
    y_count = 0

    for j in axes(g¹²ᶠᶜᵃ, 2), i in axes(g¹²ᶠᶜᵃ, 1)
        if i - 1 in axes(g¹²ᶜᶜᵃ, 1) && i in axes(g¹²ᶜᶜᵃ, 1) && j in axes(g¹²ᶜᶜᵃ, 2)
            expected_g¹²ᶠᶜᵃ = half * (g¹²ᶜᶜᵃ[i-1, j] + g¹²ᶜᶜᵃ[i, j])
            expected_G¹²ᶠᶜᵃ = half * (G¹²ᶜᶜᵃ[i-1, j] + G¹²ᶜᶜᵃ[i, j])

            max_g¹²ᶠᶜᵃ_error = max(max_g¹²ᶠᶜᵃ_error, abs(g¹²ᶠᶜᵃ[i, j] - expected_g¹²ᶠᶜᵃ))
            max_G¹²ᶠᶜᵃ_error = max(max_G¹²ᶠᶜᵃ_error, abs(G¹²ᶠᶜᵃ[i, j] - expected_G¹²ᶠᶜᵃ))
            x_count += 1
        end
    end

    for j in axes(g²¹ᶜᶠᵃ, 2), i in axes(g²¹ᶜᶠᵃ, 1)
        if i in axes(g¹²ᶜᶜᵃ, 1) && j - 1 in axes(g¹²ᶜᶜᵃ, 2) && j in axes(g¹²ᶜᶜᵃ, 2)
            expected_g²¹ᶜᶠᵃ = half * (g¹²ᶜᶜᵃ[i, j-1] + g¹²ᶜᶜᵃ[i, j])
            expected_G²¹ᶜᶠᵃ = half * (G¹²ᶜᶜᵃ[i, j-1] + G¹²ᶜᶜᵃ[i, j])

            max_g²¹ᶜᶠᵃ_error = max(max_g²¹ᶜᶠᵃ_error, abs(g²¹ᶜᶠᵃ[i, j] - expected_g²¹ᶜᶠᵃ))
            max_G²¹ᶜᶠᵃ_error = max(max_G²¹ᶜᶠᵃ_error, abs(G²¹ᶜᶠᵃ[i, j] - expected_G²¹ᶜᶠᵃ))
            y_count += 1
        end
    end

    return (; max_g¹²ᶠᶜᵃ_error,
              max_G¹²ᶠᶜᵃ_error,
              max_g²¹ᶜᶠᵃ_error,
              max_G²¹ᶜᶠᵃ_error,
              x_count,
              y_count)
end

@testset "OctaHEALPix cross-metric averaging" begin
    for FT in (Float32, Float64)
        errors = octahealpix_cross_metric_averaging_errors(FT, 8)
        tolerance = 100eps(FT)

        @test errors.x_count > 0
        @test errors.y_count > 0
        @test errors.max_g¹²ᶠᶜᵃ_error ≤ tolerance
        @test errors.max_G¹²ᶠᶜᵃ_error ≤ tolerance
        @test errors.max_g²¹ᶜᶠᵃ_error ≤ tolerance
        @test errors.max_G²¹ᶜᶠᵃ_error ≤ tolerance
    end
end
