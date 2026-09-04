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

        # Build sub-stencils (left-bias ordering, matching S₀ₙ … S₍ₙ₋₁₎ₙ)
        ψ_f64 = ntuple(Val(buffer)) do k
            start = buffer - k + 1
            ntuple(j -> S_f64[start + j - 1], Val(buffer))
        end

        ψ_f32 = ntuple(Val(buffer)) do k
            start = buffer - k + 1
            ntuple(j -> S_f32[start + j - 1], Val(buffer))
        end

        scheme_f64 = WENO(Float64; order, weight_computation=Oceananigans.Utils.NormalDivision)
        scheme_f32 = WENO(Float32; order, weight_computation=Oceananigans.Utils.NormalDivision)

        β_f64 = beta_loop(scheme_f64, ψ_f64)
        β_f32 = beta_loop(scheme_f32, ψ_f32)

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
            ω_f64 = biased_weno_weights(ψ_f64, nothing, scheme_f64)
            ω_f32 = biased_weno_weights(ψ_f32, nothing, scheme_f32)

            @test sum(ω_f64) ≈ 1
            @test sum(ω_f32) ≈ 1

            for r in 1:buffer
                @test ω_f32[r] ≈ ω_f64[r] atol=1e-3
            end
        end
    end
end
