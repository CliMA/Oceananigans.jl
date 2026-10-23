include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans: fully_supported_float_types
using Oceananigans.Advection: beta_loop, biased_weno_weights, global_smoothness_indicator, weno_reconstruction,
                              zweno_regularization, C★, ϵ
using Oceananigans.Utils: NormalDivision, BackendOptimizedDivision

# ZWENO weights evaluated in BigFloat from the smoothness indicators of `scheme`, so that only the α computation
# is compared and not the precision of β, which carries 8 significant bits in BFloat16. The regularization ϵ★
# defaults to that of `scheme`, which in formats with 8 exponent bits grows with τ beside large jumps.
function reference_weno_weights(scheme, β, τ, ϵ★ = zweno_regularization(scheme, τ))
    α = ntuple(r -> big(C★(scheme, Val(r - 1))) * (1 + (big(τ) / (big(β[r]) + big(ϵ★)))^2), length(β))
    return α ./ sum(α)
end

# A ramp of slope `jump` from the anchor over a linear background of slope `slope`
jump_stencil(FT, buffer, slope, jump) = ntuple(i -> FT(slope * (i - 1) + jump * max(0, i - buffer)), 2buffer - 1)

# A smooth profile with a large mean, ρθ ≈ 300 with O(0.1) perturbations, which rounds to a constant in BFloat16
smooth_stencil(FT, buffer) = ntuple(i -> FT(300 + 0.1 * sinpi((i - 1) / buffer)), 2buffer - 1)

@testset "WENO weights beside a jump and on a smooth profile [$FT]" for FT in fully_supported_float_types
    # The jumps span τ / (β + ϵ) from O(1) up to ≈ 2¹²⁰, whose square overflows every 8-exponent-bit format
    # but never Float64 or BigFloat, which use the unscaled weights.
    # A background slope keeps every β large enough that the rescaling thresholds themselves overflow.
    # Third order matters because it is the fallback near immersed boundaries.
    for order in (3, 5, 7, 9)
        buffer = (order + 1) ÷ 2
        stencils = [jump_stencil(FT, buffer, slope, jump) for slope in (0, 1000) for jump in exp2.(8:8:56)]
        push!(stencils, smooth_stencil(FT, buffer))

        for S in stencils, weight_computation in (NormalDivision, BackendOptimizedDivision)
            δ = ntuple(i -> S[i+1] - S[i], Val(2buffer - 2))
            scheme = WENO(FT; order, weight_computation)
            ω = biased_weno_weights(δ, nothing, scheme)

            β = beta_loop(scheme, δ)
            τ = global_smoothness_indicator(Val(buffer), β)
            reference = reference_weno_weights(scheme, β, τ)

            @test all(>=(0), β)
            @test all(isfinite, ω)
            @test sum(ω) ≈ 1
            # subnormal weights are imprecise, and cannot influence the reconstruction
            @test all(isapprox.(ω, reference; rtol=20eps(eltype(ω)), atol=floatmin(eltype(ω))))

            # the whole chain against its BigFloat counterpart with the same regularization: the FT-rounded
            # smoothness coefficients move the weights by O(eps(FT)), about 10 ulps in practice
            big_scheme = WENO(BigFloat; order, weight_computation)
            βᵇ = beta_loop(big_scheme, big.(δ))
            τᵇ = global_smoothness_indicator(Val(buffer), βᵇ)
            reference = reference_weno_weights(big_scheme, βᵇ, τᵇ, zweno_regularization(scheme, FT(τᵇ)))
            @test all(isapprox.(ω, reference; rtol=30eps(FT), atol=floatmin(eltype(ω))))

            # The regularization only redistributes weight among stencils that are smooth to within 2⁻⁶² of τ,
            # so the reconstruction agrees with the unregularized BigFloat one to the precision of FT,
            # relative to the magnitude of the stencil values
            ψ̂ = weno_reconstruction(scheme, S[buffer], δ, ω)
            ψ̂ᵇ = weno_reconstruction(big_scheme, big(S[buffer]), big.(δ), biased_weno_weights(big.(δ), nothing, big_scheme))
            @test abs(ψ̂ - ψ̂ᵇ) ≤ 10eps(FT) * maximum(abs, big.(S))
        end
    end
end

@testset "WENO weights where the flow is smooth [$FT]" for FT in fully_supported_float_types
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
