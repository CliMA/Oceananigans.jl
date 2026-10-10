include(joinpath(@__DIR__, "..", "setup", "dependencies_for_runtests.jl"))

using Oceananigans: fully_supported_float_types
using Oceananigans.Advection: beta_loop, biased_weno_weights, global_smoothness_indicator, C★, ϵ
using Oceananigans.Advection: biased_interpolate_xᶠᵃᵃ, _biased_interpolate_zᵃᵃᶠ, LeftBias, RightBias, BFloat16
using Oceananigans.Utils: NormalDivision, BackendOptimizedDivision

# ZWENO weights evaluated in BigFloat from the smoothness indicators of `scheme`, so that only the α computation
# is compared and not the precision of β, which carries 8 significant bits in BFloat16
function reference_weno_weights(scheme, β, τ)
    α = ntuple(r -> big(C★(scheme, Val(r - 1))) * (1 + (big(τ) / (big(β[r]) + big(ϵ)))^2), length(β))
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

            # the whole chain against its BigFloat counterpart: the FT-rounded smoothness coefficients
            # move the weights by O(eps(FT)), about 10 ulps in practice
            reference = biased_weno_weights(big.(δ), nothing, WENO(BigFloat; order, weight_computation))
            @test all(isapprox.(ω, reference; rtol=30eps(FT), atol=floatmin(eltype(ω))))
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

# BFloat16 is a storage format for WENO: the reconstruction of a BFloat16 field is computed and returned in Float32
@testset "BFloat16 WENO reconstruction is the Float32 reconstruction" begin
    for order in (3, 5, 7, 9), bias in (LeftBias, RightBias), weight_computation in (NormalDivision, BackendOptimizedDivision)
        buffer = (order + 1) ÷ 2
        profiles = [jump_stencil(BFloat16, buffer, slope, jump) for slope in (0, 1000) for jump in exp2.(8:8:56)]
        push!(profiles, smooth_stencil(BFloat16, buffer))

        for S in profiles
            ψ = reshape(collect((S..., S[end])), :, 1, 1)
            i = buffer + 1
            ψ̂ = biased_interpolate_xᶠᵃᵃ(i, 1, 1, nothing, WENO(BFloat16; order, weight_computation), bias, ψ)
            ψ̂³² = biased_interpolate_xᶠᵃᵃ(i, 1, 1, nothing, WENO(Float32; order, weight_computation), bias, Float32.(ψ))
            @test ψ̂ isa Float32
            @test isfinite(ψ̂)
            @test isapprox(ψ̂, ψ̂³²; rtol=2eps(Float32)) # @muladd may contract differently
        end
    end
end

# The boundary fallback (Centered in BFloat16) is promoted to the Float32 type of the WENO reconstruction
@testset "BFloat16 WENO interpolation beside a boundary is type-stable" begin
    grid = RectilinearGrid(CPU(), BFloat16; size=12, z=(0, 1), halo=5, topology=(Flat, Flat, Bounded))
    c = CenterField(grid)
    set!(c, z -> z^2)

    for order in (3, 5, 7, 9), bias in (LeftBias, RightBias), k in 1:13
        scheme = WENO(BFloat16; order, weight_computation=NormalDivision)
        @test @inferred(_biased_interpolate_zᵃᵃᶠ(1, 1, k, grid, scheme, bias, c)) isa Float32
    end
end
