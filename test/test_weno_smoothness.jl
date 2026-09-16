include("dependencies_for_runtests.jl")

using Oceananigans.Advection: beta_loop, biased_weno_weights

@testset "Float32 WENO smoothness indicators" begin
    # A smooth ρθ profile with large mean (≈ 300) and small O(0.1) perturbations.
    # This is the scenario that triggers catastrophic cancellation in naive Float32
    # β computation: the quadratic form accumulates terms ~ 300² ≈ 9e4 that must
    # cancel to leave a residual ~ 0.01, exceeding Float32's ~7-digit precision.
    for order in (5, 7, 9)
        buffer = Int((order + 1) ÷ 2)
        n_stencil = 2 * buffer  # full stencil width

        # Sample a smooth sinusoidal field: ρθ(x) = 300 + 0.1 sin(2π x)
        S_f64 = ntuple(i -> 300.0 + 0.1 * sinpi(2 * (i - 1) / n_stencil), n_stencil)
        S_f32 = ntuple(i -> Float32(S_f64[i]), n_stencil)

        # First differences of the left-bias-ordered stencil, which is S[1] … S[2buffer - 1]
        δ_f64 = ntuple(i -> S_f64[i+1] - S_f64[i], Val(2buffer - 2))
        δ_f32 = ntuple(i -> S_f32[i+1] - S_f32[i], Val(2buffer - 2))

        scheme_f64 = WENO(Float64; order, weight_computation=Oceananigans.Utils.NormalDivision)
        scheme_f32 = WENO(Float32; order, weight_computation=Oceananigans.Utils.NormalDivision)

        β_f64 = beta_loop(scheme_f64, δ_f64)
        β_f32 = beta_loop(scheme_f32, δ_f32)

        @info "WENO order $order β (Float64): $β_f64"
        @info "WENO order $order β (Float32): $β_f32"

        @testset "WENO order $order" begin
            # All Float32 β values must be non-negative
            # (negative β was the symptom of catastrophic cancellation)
            for r in 1:buffer
                @test β_f32[r] >= 0
            end

            # Float32 β should approximate Float64 reference
            for r in 1:buffer
                if β_f64[r] > 0
                    @test β_f32[r] ≈ β_f64[r] rtol=1e-2
                end
            end

            # Weights must sum to 1 and match Float64 reference
            ω_f64 = biased_weno_weights(δ_f64, nothing, scheme_f64)
            ω_f32 = biased_weno_weights(δ_f32, nothing, scheme_f32)

            @test sum(ω_f64) ≈ 1
            @test sum(ω_f32) ≈ 1

            for r in 1:buffer
                @test ω_f32[r] ≈ ω_f64[r] atol=1e-3
            end
        end
    end
end

@testset "Float32 WENO weights beside a large jump" begin
    # β and τ scale with the square of the field while ϵ is absolute, so a flat sub-stencil beside
    # a large jump drives τ / (βᵣ + ϵ) high enough that its square overflows Float32: here β ≈ 3e11
    # against ϵ = 1e-8, so the ratio is ≈ 3e19 and α = Inf without the rescaling.
    #
    # The α weights are divided by the largest such ratio, which is a common factor and so leaves
    # the normalized weights alone. The Float64 evaluation, where nothing overflows, is the
    # reference for what they should be, and the tolerance is far below Float32 round-off because
    # the rescaling introduces no error of its own.
    for order in (5, 7, 9)
        buffer = Int((order + 1) ÷ 2)
        S = ntuple(i -> i < buffer + 1 ? 0f0 : 3f5 * (i - buffer), 2buffer - 1)
        δ = ntuple(i -> S[i+1] - S[i], Val(2buffer - 2))

        for weight_computation in (Oceananigans.Utils.NormalDivision,
                                   Oceananigans.Utils.BackendOptimizedDivision)
            ω = biased_weno_weights(δ, nothing, WENO(Float32; order, weight_computation))
            reference = biased_weno_weights(Float64.(δ), nothing,
                                            WENO(Float64; order, weight_computation))

            @test all(isfinite, ω)
            @test sum(ω) ≈ 1
            @test all(isapprox.(Float64.(ω), reference; atol = 1.0e-12))
        end
    end
end
