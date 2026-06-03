using Test
using Oceananigans

# Verify fundamental metric invariants on OctaHEALPix grids at the cell
# centre (`ccc`) location, where the full covariant and contravariant
# 2x2 tensors are stored:
#   §3.3  inverse consistency g_ik G^kj = J δ_i^j
#   §3.4  positive metric determinant g_11 g_22 - g_12^2 > 0
# These are the basic algebraic invariants the metric tensor must satisfy.
# Face locations only store partial components and are covered by
# `test_octahealpix_cross_metrics.jl`.

function octahealpix_metric_invariants(FT, N)
    grid = SphericalShellGrid(CPU(), FT;
                              mapping = OctaHEALPixMapping(N),
                              z = (zero(FT), one(FT)),
                              radius = one(FT),
                              halo = (5, 5, 3))
    m = grid.metrics

    max_residual_11 = zero(FT)
    max_residual_12 = zero(FT)
    max_residual_21 = zero(FT)
    max_residual_22 = zero(FT)
    min_determinant = typemax(FT)

    for j in axes(m.g₁₁ᶜᶜᵃ, 2), i in axes(m.g₁₁ᶜᶜᵃ, 1)
        g11, g12, g22 = m.g₁₁ᶜᶜᵃ[i, j], m.g₁₂ᶜᶜᵃ[i, j], m.g₂₂ᶜᶜᵃ[i, j]
        G11, G12, G22 = m.G¹¹ᶜᶜᵃ[i, j], m.G¹²ᶜᶜᵃ[i, j], m.G²²ᶜᶜᵃ[i, j]
        J = m.Jᶜᶜᵃ[i, j]

        # g_ik G^kj should equal J δ_i^j (since G = J g^{-1}).
        m11 = g11 * G11 + g12 * G12
        m12 = g11 * G12 + g12 * G22
        m21 = g12 * G11 + g22 * G12
        m22 = g12 * G12 + g22 * G22

        max_residual_11 = max(max_residual_11, abs(m11 - J))
        max_residual_12 = max(max_residual_12, abs(m12))
        max_residual_21 = max(max_residual_21, abs(m21))
        max_residual_22 = max(max_residual_22, abs(m22 - J))

        determinant = g11 * g22 - g12^2
        min_determinant = min(min_determinant, determinant)
    end

    return (; max_residual_11, max_residual_12, max_residual_21, max_residual_22,
              min_determinant)
end

@testset "OctaHEALPix metric invariants (CCC)" begin
    for FT in (Float32, Float64)
        result = octahealpix_metric_invariants(FT, 8)
        tolerance = 100eps(FT)

        # g·G = J·I to roundoff (residual scaled to J ≈ O(1) on unit sphere).
        @test result.max_residual_11 ≤ tolerance
        @test result.max_residual_12 ≤ tolerance
        @test result.max_residual_21 ≤ tolerance
        @test result.max_residual_22 ≤ tolerance

        # Positive metric determinant everywhere (non-degenerate cells).
        @test result.min_determinant > zero(FT)
    end
end
