include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans.Advection: beta_loop, biased_weno_weights, global_smoothness_indicator, C★, ϵ
using Oceananigans.Utils: NormalDivision, BackendOptimizedDivision
using BFloat16s: BFloat16

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

        @testset "WENO order $order" begin
            @testset let β_f32=β_f32, β_f64=β_f64
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
                    @test ω_f32[r] ≈ ω_f64[r] rtol=2e-3
                end
            end
        end
    end
end

# ZWENO weights evaluated in BigFloat from the smoothness indicators of `scheme`, so that only the α computation
# is compared and not the precision of β, which carries 8 significant bits in BFloat16
function reference_weno_weights(scheme, β, τ)
    α = ntuple(r -> big(C★(scheme, Val(r - 1))) * (1 + (big(τ) / (big(β[r]) + big(ϵ)))^2), length(β))
    return α ./ sum(α)
end

const weno_float_types = (Float64, Float32, BFloat16, BigFloat)

jump_stencil(FT, buffer, slope, jump) = ntuple(i -> FT(slope * (i - 1) + jump * max(0, i - buffer)), 2buffer - 1)

@testset "WENO weights beside a large jump [$FT]" for FT in weno_float_types
    # The jumps span τ / (β + ϵ) from O(1) up to ≈ 2¹²⁰, whose square overflows every 8-exponent-bit format
    # but never Float64 or BigFloat, which use the unscaled weights.
    # A background slope keeps every β large enough that the rescaling thresholds themselves overflow.
    # Third order matters because it is the fallback near immersed boundaries.
    for order in (3, 5, 7, 9), slope in (0, 1000), jump in exp2.(8:8:56)
        buffer = (order + 1) ÷ 2
        S = jump_stencil(FT, buffer, slope, jump)
        δ = ntuple(i -> S[i+1] - S[i], Val(2buffer - 2))

        for weight_computation in (NormalDivision, BackendOptimizedDivision)
            scheme = WENO(FT; order, weight_computation)
            ω = biased_weno_weights(δ, nothing, scheme)

            β = beta_loop(scheme, δ)
            τ = global_smoothness_indicator(Val(buffer), β)
            reference = reference_weno_weights(scheme, β, τ)

            @test all(isfinite, ω)
            @test sum(ω) ≈ 1
            # subnormal weights are imprecise, and cannot influence the reconstruction
            @test all(isapprox.(ω, reference; rtol=100eps(eltype(ω)), atol=floatmin(eltype(ω))))

            # the whole chain against its BigFloat counterpart: the FT-rounded smoothness coefficients
            # move the weights by O(eps(FT)), about 10 ulps in practice
            reference = biased_weno_weights(big.(δ), nothing, WENO(BigFloat; order, weight_computation))
            @test all(isapprox.(ω, reference; rtol=50eps(FT), atol=floatmin(eltype(ω))))
        end
    end
end

@testset "WENO weights where the flow is smooth [$FT]" for FT in weno_float_types
    for order in (3, 5, 7, 9)
        buffer = (order + 1) ÷ 2
        δ = ntuple(_ -> one(FT), Val(2buffer - 2)) # linear field ⇒ every β equal ⇒ τ = 0

        for weight_computation in (NormalDivision, BackendOptimizedDivision)
            scheme = WENO(FT; order, weight_computation)
            ω = biased_weno_weights(δ, nothing, scheme)
            optimal = ntuple(r -> big(C★(scheme, Val(r - 1))), buffer)
            optimal = optimal ./ sum(optimal) # the FT-rounded C★ do not sum exactly to one

            β = beta_loop(scheme, δ)
            τ = global_smoothness_indicator(Val(buffer), β)

            @test all(isfinite, ω)
            @test sum(ω) ≈ 1
            @test all(isapprox.(ω, reference_weno_weights(scheme, β, τ); rtol=100eps(eltype(ω))))
            # the β agree only to FT rounding, so τ ≠ 0 moves the weights off the optimal ones by O((τ / β)²)
            @test all(isapprox.(ω, optimal; rtol=1e-6 + (τ / minimum(β))^2))
        end
    end
end
