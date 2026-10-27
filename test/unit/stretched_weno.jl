include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Advection: biased_coefficients, weno_reconstruction, coeff_p, smoothness_coefficients, C★,
                              materialize_advection, biased_interpolate_xᶠᶜᶜ, LeftBias

function stretched_reconstruction_error(order, stretched, Nx)
    stretching(ξ) = ξ + 7 * sin(π * ξ) / 20π
    primitive(x) = -cos(2π * x) / 2π
    grid = RectilinearGrid(CPU(); size = Nx, x = i -> stretching((i - 1) / Nx), halo = 6, topology = (Bounded, Flat, Flat))
    scheme = materialize_advection(WENO(Float64; order, stretched), grid)
    xᶠ = xnodes(grid, Face())
    c = CenterField(grid)
    set!(c, reshape([(primitive(xᶠ[i+1]) - primitive(xᶠ[i])) / (xᶠ[i+1] - xᶠ[i]) for i in 1:Nx], Nx, 1, 1))
    return maximum(abs(biased_interpolate_xᶠᶜᶜ(i, 1, 1, grid, scheme, LeftBias, c) - sin(2π * xᶠ[i])) for i in order+1:Nx-order+1)
end

@testset "Stretched WENO" begin
    @testset "Uniform cells give the constant coefficients [WENO$(2N - 1)]" for N in 2:5
        coefficients = biased_coefficients(Float64, N, 0:2N-1)
        scheme = WENO(Float64; order = 2N - 1)

        for s in 0:N-1
            @test all(isapprox.(coefficients.p[s+1], coeff_p(scheme, Val(s)), atol = 1e-14))
            @test all(isapprox.(coefficients.β[s+1], smoothness_coefficients(Val(Float64), Val(N), Val(s)), atol = 1e-14))
            @test coefficients.C★[s+1] ≈ C★(scheme, Val(s)) atol = 1e-14
        end
    end

    @testset "Polynomials on geometric cells are reconstructed exactly [WENO$(2N - 1)]" for N in (3, 4)
        edges = cumsum([0.0; [1.07^m for m in 1:2N-1]])
        coefficients = biased_coefficients(Float64, N, edges)
        scheme = WENO(Float64; order = 2N - 1)

        for degree in 0:2N-2
            average(v) = (edges[v+1]^(degree+1) - edges[v]^(degree+1)) / ((degree + 1) * (edges[v+1] - edges[v]))
            S = Tuple(average(v) for v in 1:2N-1)
            δ = Tuple(S[m+1] - S[m] for m in 1:2N-2)
            @test weno_reconstruction(scheme, S[N], δ, coefficients.C★, coefficients) ≈ edges[N+1]^degree rtol = 1e-12
        end
    end

    @testset "Order of the face reconstruction on a stretched grid" begin
        uniform = [stretched_reconstruction_error(5, false, Nx) for Nx in (64, 128)]
        stretched = [stretched_reconstruction_error(5, true, Nx) for Nx in (64, 128)]
        @test log2(uniform[1] / uniform[2]) < 2.2
        @test log2(stretched[1] / stretched[2]) > 4.5
        @test stretched[2] < uniform[2] / 1000
    end

    @testset "Selected directions" begin
        stretching(i) = (ξ = (i - 1) / 8; ξ + 7 * sin(π * ξ) / 20π)
        grid = RectilinearGrid(CPU(); size = (8, 8, 8), x = stretching, y = stretching, z = stretching, halo = (3, 3, 3))
        tabulated(scheme) = map(!isnothing, values(materialize_advection(scheme, grid).coefficients))

        @test tabulated(WENO(order = 5, stretched = true)) == (true, true, true)
        @test tabulated(WENO(order = 5, stretched = (:z,))) == (false, false, true)
        @test tabulated(WENO(order = 5, stretched = (:x, :z))) == (true, false, true)
        @test isnothing(materialize_advection(WENO(order = 5), grid).coefficients)
        @test_throws ArgumentError WENO(order = 5, stretched = (:w,))
    end

    @testset "Summary" begin
        @test summary(WENO(order = 5, stretched = true)) == "WENO{3, Float64, Nothing}(order=5, stretched)"
        @test summary(WENO(order = 5)) == "WENO{3, Float64, Nothing}(order=5)"
    end
end
