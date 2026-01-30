#####
##### Low-register WENO interpolation using sliding window approach
##### for VelocityStencil and FunctionStencil cases
#####
##### Instead of loading all stencils at once (high register pressure), we:
##### 1. Load initial window of `buffer` values for both reconstruction (ψ) and smoothness (u/v or func)
##### 2. For each stencil: compute β from smoothness stencil and p from reconstruction stencil, then slide windows
##### 3. Compute weights from all β values
##### 4. Return weighted sum of biased_p values
#####
##### This reduces register usage from O(buffer²) to O(buffer)
#####

using Oceananigans.Operators: ℑyᵃᶠᵃ, ℑxᶠᵃᵃ

#####
##### Remove `DefaultStencil` implementations (same as normal WENO)
#####

for N in [2, 3, 4, 5, 6]
    @eval begin
        fused_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme::WENO{$N}, bias, ψ, ::DefaultStencil, args...) =
            fused_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, scheme, bias, ψ, args...)
        
        fused_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme::WENO{$N}, bias, ψ, ::DefaultStencil, args...) =
            fused_biased_interpolate_yᵃᶠᵃ(i, j, k, grid, scheme, bias, ψ, args...) =
        
        fused_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme::WENO{$N}, bias, ψ, ::DefaultStencil, args...) =
            fused_biased_interpolate_zᵃᵃᶠ(i, j, k, grid, scheme, bias, ψ, args...)
    end
end

#####
##### Fused interpolation for VelocityStencil - x direction
#####

@inline function fused_biased_interpolate_xᶠᵃᵃ(i, j, k, grid,
                                               scheme::WENO{2, FT}, bias,
                                               ψ, ::VelocityStencil, u, v, args...) where FT
    # Compute all β values first (reusing registers for u and v)
    w1 = ℑyᵃᶠᵃ(i + stencil_offset(bias, 0, 0), j, k, grid, u)
    w2 = ℑyᵃᶠᵃ(i + stencil_offset(bias, 0, 1), j, k, grid, u)
    βu1 = smoothness_indicator((w1, w2), scheme, Val(0))

    w2 = w1
    w1 = ℑyᵃᶠᵃ(i + stencil_offset(bias, 1, 0), j, k, grid, u)
    βu2 = smoothness_indicator((w1, w2), scheme, Val(1))

    # Reuse same registers for v
    w1 = ℑxᶠᵃᵃ(i + stencil_offset(bias, 0, 0), j, k, grid, v)
    w2 = ℑxᶠᵃᵃ(i + stencil_offset(bias, 0, 1), j, k, grid, v)
    βv1 = smoothness_indicator((w1, w2), scheme, Val(0))

    w2 = w1
    w1 = ℑxᶠᵃᵃ(i + stencil_offset(bias, 1, 0), j, k, grid, v)
    βv2 = smoothness_indicator((w1, w2), scheme, Val(1))

    β1 = (βu1 + βv1) / 2
    β2 = (βu2 + βv2) / 2

    # Compute weights from β values
    τ = global_smoothness_indicator(Val(2), (β1, β2))
    α = zweno_alpha_loop(scheme, (β1, β2), τ)
    Σα = α[1] + α[2]

    # Now compute all p values (reusing same registers for w)
    w1 = getvalue(ψ, i + stencil_offset(bias, 0, 0), j, k, grid, u, v, args...)
    w2 = getvalue(ψ, i + stencil_offset(bias, 0, 1), j, k, grid, u, v, args...)
    p1 = biased_p(scheme, bias, Val(0), (w1, w2))

    w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 1, 0), j, k, grid, u, v, args...)
    p2 = biased_p(scheme, bias, Val(1), (w1, w2))

    return @muladd (α[1] * p1 + α[2] * p2) / Σα
end

@inline function fused_biased_interpolate_xᶠᵃᵃ(i, j, k, grid,
                                               scheme::WENO{3, FT}, bias,
                                               ψ, ::VelocityStencil, u, v, args...) where FT
    # Compute all β values first (reusing registers for u and v)
    w1 = ℑyᵃᶠᵃ(i + stencil_offset(bias, 0, 0), j, k, grid, u)
    w2 = ℑyᵃᶠᵃ(i + stencil_offset(bias, 0, 1), j, k, grid, u)
    w3 = ℑyᵃᶠᵃ(i + stencil_offset(bias, 0, 2), j, k, grid, u)
    βu1 = smoothness_indicator((w1, w2, w3), scheme, Val(0))

    w3 = w2; w2 = w1
    w1 = ℑyᵃᶠᵃ(i + stencil_offset(bias, 1, 0), j, k, grid, u)
    βu2 = smoothness_indicator((w1, w2, w3), scheme, Val(1))

    w3 = w2; w2 = w1
    w1 = ℑyᵃᶠᵃ(i + stencil_offset(bias, 2, 0), j, k, grid, u)
    βu3 = smoothness_indicator((w1, w2, w3), scheme, Val(2))

    # Reuse same registers for v
    w1 = ℑxᶠᵃᵃ(i + stencil_offset(bias, 0, 0), j, k, grid, v)
    w2 = ℑxᶠᵃᵃ(i + stencil_offset(bias, 0, 1), j, k, grid, v)
    w3 = ℑxᶠᵃᵃ(i + stencil_offset(bias, 0, 2), j, k, grid, v)
    βv1 = smoothness_indicator((w1, w2, w3), scheme, Val(0))

    w3 = w2; w2 = w1
    w1 = ℑxᶠᵃᵃ(i + stencil_offset(bias, 1, 0), j, k, grid, v)
    βv2 = smoothness_indicator((w1, w2, w3), scheme, Val(1))

    w3 = w2; w2 = w1
    w1 = ℑxᶠᵃᵃ(i + stencil_offset(bias, 2, 0), j, k, grid, v)
    βv3 = smoothness_indicator((w1, w2, w3), scheme, Val(2))

    β1 = (βu1 + βv1) / 2
    β2 = (βu2 + βv2) / 2
    β3 = (βu3 + βv3) / 2

    # Compute weights from β values
    τ = global_smoothness_indicator(Val(3), (β1, β2, β3))
    α = zweno_alpha_loop(scheme, (β1, β2, β3), τ)
    Σα = α[1] + α[2] + α[3]

    # Now compute all p values (reusing same registers for w)
    w1 = getvalue(ψ, i + stencil_offset(bias, 0, 0), j, k, grid, u, v, args...)
    w2 = getvalue(ψ, i + stencil_offset(bias, 0, 1), j, k, grid, u, v, args...)
    w3 = getvalue(ψ, i + stencil_offset(bias, 0, 2), j, k, grid, u, v, args...)
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3))

    w3 = w2; w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 1, 0), j, k, grid, u, v, args...)
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3))

    w3 = w2; w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 2, 0), j, k, grid, u, v, args...)
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3))

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3) / Σα
end

@inline function fused_biased_interpolate_xᶠᵃᵃ(i, j, k, grid,
                                               scheme::WENO{4, FT}, bias,
                                               ψ, ::VelocityStencil, u, v, args...) where FT
    # Compute all β values first (reusing registers for u and v)
    w1 = ℑyᵃᶠᵃ(i + stencil_offset(bias, 0, 0), j, k, grid, u)
    w2 = ℑyᵃᶠᵃ(i + stencil_offset(bias, 0, 1), j, k, grid, u)
    w3 = ℑyᵃᶠᵃ(i + stencil_offset(bias, 0, 2), j, k, grid, u)
    w4 = ℑyᵃᶠᵃ(i + stencil_offset(bias, 0, 3), j, k, grid, u)
    βu1 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(0))

    w4 = w3; w3 = w2; w2 = w1
    w1 = ℑyᵃᶠᵃ(i + stencil_offset(bias, 1, 0), j, k, grid, u)
    βu2 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(1))

    w4 = w3; w3 = w2; w2 = w1
    w1 = ℑyᵃᶠᵃ(i + stencil_offset(bias, 2, 0), j, k, grid, u)
    βu3 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(2))

    w4 = w3; w3 = w2; w2 = w1
    w1 = ℑyᵃᶠᵃ(i + stencil_offset(bias, 3, 0), j, k, grid, u)
    βu4 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(3))

    # Reuse same registers for v
    w1 = ℑxᶠᵃᵃ(i + stencil_offset(bias, 0, 0), j, k, grid, v)
    w2 = ℑxᶠᵃᵃ(i + stencil_offset(bias, 0, 1), j, k, grid, v)
    w3 = ℑxᶠᵃᵃ(i + stencil_offset(bias, 0, 2), j, k, grid, v)
    w4 = ℑxᶠᵃᵃ(i + stencil_offset(bias, 0, 3), j, k, grid, v)
    βv1 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(0))

    w4 = w3; w3 = w2; w2 = w1
    w1 = ℑxᶠᵃᵃ(i + stencil_offset(bias, 1, 0), j, k, grid, v)
    βv2 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(1))

    w4 = w3; w3 = w2; w2 = w1
    w1 = ℑxᶠᵃᵃ(i + stencil_offset(bias, 2, 0), j, k, grid, v)
    βv3 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(2))

    w4 = w3; w3 = w2; w2 = w1
    w1 = ℑxᶠᵃᵃ(i + stencil_offset(bias, 3, 0), j, k, grid, v)
    βv4 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(3))

    β1 = (βu1 + βv1) / 2
    β2 = (βu2 + βv2) / 2
    β3 = (βu3 + βv3) / 2
    β4 = (βu4 + βv4) / 2

    # Compute weights from β values
    τ = global_smoothness_indicator(Val(4), (β1, β2, β3, β4))
    α = zweno_alpha_loop(scheme, (β1, β2, β3, β4), τ)
    Σα = α[1] + α[2] + α[3] + α[4]

    # Now compute all p values (reusing same registers for w)
    w1 = getvalue(ψ, i + stencil_offset(bias, 0, 0), j, k, grid, u, v, args...)
    w2 = getvalue(ψ, i + stencil_offset(bias, 0, 1), j, k, grid, u, v, args...)
    w3 = getvalue(ψ, i + stencil_offset(bias, 0, 2), j, k, grid, u, v, args...)
    w4 = getvalue(ψ, i + stencil_offset(bias, 0, 3), j, k, grid, u, v, args...)
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4))

    w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 1, 0), j, k, grid, u, v, args...)
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4))

    w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 2, 0), j, k, grid, u, v, args...)
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4))

    w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 3, 0), j, k, grid, u, v, args...)
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4))

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3 + α[4] * p4) / Σα
end

@inline function fused_biased_interpolate_xᶠᵃᵃ(i, j, k, grid,
                                               scheme::WENO{5, FT}, bias,
                                               ψ, ::VelocityStencil, u, v, args...) where FT
    # Compute all β values first (reusing registers for u and v)
    w1 = ℑyᵃᶠᵃ(i + stencil_offset(bias, 0, 0), j, k, grid, u)
    w2 = ℑyᵃᶠᵃ(i + stencil_offset(bias, 0, 1), j, k, grid, u)
    w3 = ℑyᵃᶠᵃ(i + stencil_offset(bias, 0, 2), j, k, grid, u)
    w4 = ℑyᵃᶠᵃ(i + stencil_offset(bias, 0, 3), j, k, grid, u)
    w5 = ℑyᵃᶠᵃ(i + stencil_offset(bias, 0, 4), j, k, grid, u)
    βu1 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(0))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ℑyᵃᶠᵃ(i + stencil_offset(bias, 1, 0), j, k, grid, u)
    βu2 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(1))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ℑyᵃᶠᵃ(i + stencil_offset(bias, 2, 0), j, k, grid, u)
    βu3 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(2))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ℑyᵃᶠᵃ(i + stencil_offset(bias, 3, 0), j, k, grid, u)
    βu4 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(3))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ℑyᵃᶠᵃ(i + stencil_offset(bias, 4, 0), j, k, grid, u)
    βu5 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(4))

    # Reuse same registers for v
    w1 = ℑxᶠᵃᵃ(i + stencil_offset(bias, 0, 0), j, k, grid, v)
    w2 = ℑxᶠᵃᵃ(i + stencil_offset(bias, 0, 1), j, k, grid, v)
    w3 = ℑxᶠᵃᵃ(i + stencil_offset(bias, 0, 2), j, k, grid, v)
    w4 = ℑxᶠᵃᵃ(i + stencil_offset(bias, 0, 3), j, k, grid, v)
    w5 = ℑxᶠᵃᵃ(i + stencil_offset(bias, 0, 4), j, k, grid, v)
    βv1 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(0))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ℑxᶠᵃᵃ(i + stencil_offset(bias, 1, 0), j, k, grid, v)
    βv2 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(1))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ℑxᶠᵃᵃ(i + stencil_offset(bias, 2, 0), j, k, grid, v)
    βv3 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(2))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ℑxᶠᵃᵃ(i + stencil_offset(bias, 3, 0), j, k, grid, v)
    βv4 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(3))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ℑxᶠᵃᵃ(i + stencil_offset(bias, 4, 0), j, k, grid, v)
    βv5 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(4))

    β1 = (βu1 + βv1) / 2
    β2 = (βu2 + βv2) / 2
    β3 = (βu3 + βv3) / 2
    β4 = (βu4 + βv4) / 2
    β5 = (βu5 + βv5) / 2

    # Compute weights from β values
    τ = global_smoothness_indicator(Val(5), (β1, β2, β3, β4, β5))
    α = zweno_alpha_loop(scheme, (β1, β2, β3, β4, β5), τ)
    Σα = α[1] + α[2] + α[3] + α[4] + α[5]

    # Now compute all p values (reusing same registers for w)
    w1 = getvalue(ψ, i + stencil_offset(bias, 0, 0), j, k, grid, u, v, args...)
    w2 = getvalue(ψ, i + stencil_offset(bias, 0, 1), j, k, grid, u, v, args...)
    w3 = getvalue(ψ, i + stencil_offset(bias, 0, 2), j, k, grid, u, v, args...)
    w4 = getvalue(ψ, i + stencil_offset(bias, 0, 3), j, k, grid, u, v, args...)
    w5 = getvalue(ψ, i + stencil_offset(bias, 0, 4), j, k, grid, u, v, args...)
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 1, 0), j, k, grid, u, v, args...)
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 2, 0), j, k, grid, u, v, args...)
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 3, 0), j, k, grid, u, v, args...)
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 4, 0), j, k, grid, u, v, args...)
    p5 = biased_p(scheme, bias, Val(4), (w1, w2, w3, w4, w5))

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3 + α[4] * p4 + α[5] * p5) / Σα
end

@inline function fused_biased_interpolate_xᶠᵃᵃ(i, j, k, grid,
                                               scheme::WENO{6, FT}, bias,
                                               ψ, ::VelocityStencil, u, v, args...) where FT
    # Compute all β values first (reusing registers for u and v)
    w1 = ℑyᵃᶠᵃ(i + stencil_offset(bias, 0, 0), j, k, grid, u)
    w2 = ℑyᵃᶠᵃ(i + stencil_offset(bias, 0, 1), j, k, grid, u)
    w3 = ℑyᵃᶠᵃ(i + stencil_offset(bias, 0, 2), j, k, grid, u)
    w4 = ℑyᵃᶠᵃ(i + stencil_offset(bias, 0, 3), j, k, grid, u)
    w5 = ℑyᵃᶠᵃ(i + stencil_offset(bias, 0, 4), j, k, grid, u)
    w6 = ℑyᵃᶠᵃ(i + stencil_offset(bias, 0, 5), j, k, grid, u)
    βu1 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(0))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ℑyᵃᶠᵃ(i + stencil_offset(bias, 1, 0), j, k, grid, u)
    βu2 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(1))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ℑyᵃᶠᵃ(i + stencil_offset(bias, 2, 0), j, k, grid, u)
    βu3 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(2))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ℑyᵃᶠᵃ(i + stencil_offset(bias, 3, 0), j, k, grid, u)
    βu4 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(3))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ℑyᵃᶠᵃ(i + stencil_offset(bias, 4, 0), j, k, grid, u)
    βu5 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(4))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ℑyᵃᶠᵃ(i + stencil_offset(bias, 5, 0), j, k, grid, u)
    βu6 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(5))

    # Reuse same registers for v
    w1 = ℑxᶠᵃᵃ(i + stencil_offset(bias, 0, 0), j, k, grid, v)
    w2 = ℑxᶠᵃᵃ(i + stencil_offset(bias, 0, 1), j, k, grid, v)
    w3 = ℑxᶠᵃᵃ(i + stencil_offset(bias, 0, 2), j, k, grid, v)
    w4 = ℑxᶠᵃᵃ(i + stencil_offset(bias, 0, 3), j, k, grid, v)
    w5 = ℑxᶠᵃᵃ(i + stencil_offset(bias, 0, 4), j, k, grid, v)
    w6 = ℑxᶠᵃᵃ(i + stencil_offset(bias, 0, 5), j, k, grid, v)
    βv1 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(0))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ℑxᶠᵃᵃ(i + stencil_offset(bias, 1, 0), j, k, grid, v)
    βv2 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(1))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ℑxᶠᵃᵃ(i + stencil_offset(bias, 2, 0), j, k, grid, v)
    βv3 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(2))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ℑxᶠᵃᵃ(i + stencil_offset(bias, 3, 0), j, k, grid, v)
    βv4 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(3))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ℑxᶠᵃᵃ(i + stencil_offset(bias, 4, 0), j, k, grid, v)
    βv5 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(4))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ℑxᶠᵃᵃ(i + stencil_offset(bias, 5, 0), j, k, grid, v)
    βv6 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(5))

    β1 = (βu1 + βv1) / 2
    β2 = (βu2 + βv2) / 2
    β3 = (βu3 + βv3) / 2
    β4 = (βu4 + βv4) / 2
    β5 = (βu5 + βv5) / 2
    β6 = (βu6 + βv6) / 2

    # Compute weights from β values
    τ = global_smoothness_indicator(Val(6), (β1, β2, β3, β4, β5, β6))
    α = zweno_alpha_loop(scheme, (β1, β2, β3, β4, β5, β6), τ)
    Σα = α[1] + α[2] + α[3] + α[4] + α[5] + α[6]

    # Now compute all p values (reusing same registers for w)
    w1 = getvalue(ψ, i + stencil_offset(bias, 0, 0), j, k, grid, u, v, args...)
    w2 = getvalue(ψ, i + stencil_offset(bias, 0, 1), j, k, grid, u, v, args...)
    w3 = getvalue(ψ, i + stencil_offset(bias, 0, 2), j, k, grid, u, v, args...)
    w4 = getvalue(ψ, i + stencil_offset(bias, 0, 3), j, k, grid, u, v, args...)
    w5 = getvalue(ψ, i + stencil_offset(bias, 0, 4), j, k, grid, u, v, args...)
    w6 = getvalue(ψ, i + stencil_offset(bias, 0, 5), j, k, grid, u, v, args...)
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 1, 0), j, k, grid, u, v, args...)
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 2, 0), j, k, grid, u, v, args...)
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 3, 0), j, k, grid, u, v, args...)
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 4, 0), j, k, grid, u, v, args...)
    p5 = biased_p(scheme, bias, Val(4), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 5, 0), j, k, grid, u, v, args...)
    p6 = biased_p(scheme, bias, Val(5), (w1, w2, w3, w4, w5, w6))

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3 + α[4] * p4 + α[5] * p5 + α[6] * p6) / Σα
end

#####
##### Fused interpolation for FunctionStencil - x direction
#####

@inline function fused_biased_interpolate_xᶠᵃᵃ(i, j, k, grid,
                                               scheme::WENO{2, FT}, bias,
                                               ψ, VI::FunctionStencil, args...) where FT
    # Compute all β values first (reusing registers)
    w1 = getvalue(VI.func, i + stencil_offset(bias, 0, 0), j, k, grid, args...)
    w2 = getvalue(VI.func, i + stencil_offset(bias, 0, 1), j, k, grid, args...)
    β1 = smoothness_indicator((w1, w2), scheme, Val(0))

    w2 = w1
    w1 = getvalue(VI.func, i + stencil_offset(bias, 1, 0), j, k, grid, args...)
    β2 = smoothness_indicator((w1, w2), scheme, Val(1))

    # Compute weights from β values
    τ = global_smoothness_indicator(Val(2), (β1, β2))
    α = zweno_alpha_loop(scheme, (β1, β2), τ)
    Σα = α[1] + α[2]

    # Now compute all p values (reusing same registers)
    w1 = getvalue(ψ, i + stencil_offset(bias, 0, 0), j, k, grid, args...)
    w2 = getvalue(ψ, i + stencil_offset(bias, 0, 1), j, k, grid, args...)
    p1 = biased_p(scheme, bias, Val(0), (w1, w2))

    w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 1, 0), j, k, grid, args...)
    p2 = biased_p(scheme, bias, Val(1), (w1, w2))

    return @muladd (α[1] * p1 + α[2] * p2) / Σα
end

@inline function fused_biased_interpolate_xᶠᵃᵃ(i, j, k, grid,
                                               scheme::WENO{3, FT}, bias,
                                               ψ, VI::FunctionStencil, args...) where FT
    # Compute all β values first (reusing registers)
    w1 = getvalue(VI.func, i + stencil_offset(bias, 0, 0), j, k, grid, args...)
    w2 = getvalue(VI.func, i + stencil_offset(bias, 0, 1), j, k, grid, args...)
    w3 = getvalue(VI.func, i + stencil_offset(bias, 0, 2), j, k, grid, args...)
    β1 = smoothness_indicator((w1, w2, w3), scheme, Val(0))

    w3 = w2; w2 = w1
    w1 = getvalue(VI.func, i + stencil_offset(bias, 1, 0), j, k, grid, args...)
    β2 = smoothness_indicator((w1, w2, w3), scheme, Val(1))

    w3 = w2; w2 = w1
    w1 = getvalue(VI.func, i + stencil_offset(bias, 2, 0), j, k, grid, args...)
    β3 = smoothness_indicator((w1, w2, w3), scheme, Val(2))

    # Compute weights from β values
    τ = global_smoothness_indicator(Val(3), (β1, β2, β3))
    α = zweno_alpha_loop(scheme, (β1, β2, β3), τ)
    Σα = α[1] + α[2] + α[3]

    # Now compute all p values (reusing same registers)
    w1 = getvalue(ψ, i + stencil_offset(bias, 0, 0), j, k, grid, args...)
    w2 = getvalue(ψ, i + stencil_offset(bias, 0, 1), j, k, grid, args...)
    w3 = getvalue(ψ, i + stencil_offset(bias, 0, 2), j, k, grid, args...)
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3))

    w3 = w2; w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 1, 0), j, k, grid, args...)
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3))

    w3 = w2; w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 2, 0), j, k, grid, args...)
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3))

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3) / Σα
end

@inline function fused_biased_interpolate_xᶠᵃᵃ(i, j, k, grid,
                                               scheme::WENO{4, FT}, bias,
                                               ψ, VI::FunctionStencil, args...) where FT
    # Compute all β values first (reusing registers)
    w1 = getvalue(VI.func, i + stencil_offset(bias, 0, 0), j, k, grid, args...)
    w2 = getvalue(VI.func, i + stencil_offset(bias, 0, 1), j, k, grid, args...)
    w3 = getvalue(VI.func, i + stencil_offset(bias, 0, 2), j, k, grid, args...)
    w4 = getvalue(VI.func, i + stencil_offset(bias, 0, 3), j, k, grid, args...)
    β1 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(0))

    w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(VI.func, i + stencil_offset(bias, 1, 0), j, k, grid, args...)
    β2 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(1))

    w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(VI.func, i + stencil_offset(bias, 2, 0), j, k, grid, args...)
    β3 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(2))

    w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(VI.func, i + stencil_offset(bias, 3, 0), j, k, grid, args...)
    β4 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(3))

    # Compute weights from β values
    τ = global_smoothness_indicator(Val(4), (β1, β2, β3, β4))
    α = zweno_alpha_loop(scheme, (β1, β2, β3, β4), τ)
    Σα = α[1] + α[2] + α[3] + α[4]

    # Now compute all p values (reusing same registers)
    w1 = getvalue(ψ, i + stencil_offset(bias, 0, 0), j, k, grid, args...)
    w2 = getvalue(ψ, i + stencil_offset(bias, 0, 1), j, k, grid, args...)
    w3 = getvalue(ψ, i + stencil_offset(bias, 0, 2), j, k, grid, args...)
    w4 = getvalue(ψ, i + stencil_offset(bias, 0, 3), j, k, grid, args...)
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4))

    w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 1, 0), j, k, grid, args...)
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4))

    w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 2, 0), j, k, grid, args...)
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4))

    w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 3, 0), j, k, grid, args...)
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4))

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3 + α[4] * p4) / Σα
end

@inline function fused_biased_interpolate_xᶠᵃᵃ(i, j, k, grid,
                                               scheme::WENO{5, FT}, bias,
                                               ψ, VI::FunctionStencil, args...) where FT
    # Compute all β values first (reusing registers)
    w1 = getvalue(VI.func, i + stencil_offset(bias, 0, 0), j, k, grid, args...)
    w2 = getvalue(VI.func, i + stencil_offset(bias, 0, 1), j, k, grid, args...)
    w3 = getvalue(VI.func, i + stencil_offset(bias, 0, 2), j, k, grid, args...)
    w4 = getvalue(VI.func, i + stencil_offset(bias, 0, 3), j, k, grid, args...)
    w5 = getvalue(VI.func, i + stencil_offset(bias, 0, 4), j, k, grid, args...)
    β1 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(0))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(VI.func, i + stencil_offset(bias, 1, 0), j, k, grid, args...)
    β2 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(1))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(VI.func, i + stencil_offset(bias, 2, 0), j, k, grid, args...)
    β3 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(2))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(VI.func, i + stencil_offset(bias, 3, 0), j, k, grid, args...)
    β4 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(3))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(VI.func, i + stencil_offset(bias, 4, 0), j, k, grid, args...)
    β5 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(4))

    # Compute weights from β values
    τ = global_smoothness_indicator(Val(5), (β1, β2, β3, β4, β5))
    α = zweno_alpha_loop(scheme, (β1, β2, β3, β4, β5), τ)
    Σα = α[1] + α[2] + α[3] + α[4] + α[5]

    # Now compute all p values (reusing same registers)
    w1 = getvalue(ψ, i + stencil_offset(bias, 0, 0), j, k, grid, args...)
    w2 = getvalue(ψ, i + stencil_offset(bias, 0, 1), j, k, grid, args...)
    w3 = getvalue(ψ, i + stencil_offset(bias, 0, 2), j, k, grid, args...)
    w4 = getvalue(ψ, i + stencil_offset(bias, 0, 3), j, k, grid, args...)
    w5 = getvalue(ψ, i + stencil_offset(bias, 0, 4), j, k, grid, args...)
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 1, 0), j, k, grid, args...)
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 2, 0), j, k, grid, args...)
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 3, 0), j, k, grid, args...)
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 4, 0), j, k, grid, args...)
    p5 = biased_p(scheme, bias, Val(4), (w1, w2, w3, w4, w5))

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3 + α[4] * p4 + α[5] * p5) / Σα
end

@inline function fused_biased_interpolate_xᶠᵃᵃ(i, j, k, grid,
                                               scheme::WENO{6, FT}, bias,
                                               ψ, VI::FunctionStencil, args...) where FT
    # Compute all β values first (reusing registers)
    w1 = getvalue(VI.func, i + stencil_offset(bias, 0, 0), j, k, grid, args...)
    w2 = getvalue(VI.func, i + stencil_offset(bias, 0, 1), j, k, grid, args...)
    w3 = getvalue(VI.func, i + stencil_offset(bias, 0, 2), j, k, grid, args...)
    w4 = getvalue(VI.func, i + stencil_offset(bias, 0, 3), j, k, grid, args...)
    w5 = getvalue(VI.func, i + stencil_offset(bias, 0, 4), j, k, grid, args...)
    w6 = getvalue(VI.func, i + stencil_offset(bias, 0, 5), j, k, grid, args...)
    β1 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(0))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(VI.func, i + stencil_offset(bias, 1, 0), j, k, grid, args...)
    β2 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(1))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(VI.func, i + stencil_offset(bias, 2, 0), j, k, grid, args...)
    β3 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(2))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(VI.func, i + stencil_offset(bias, 3, 0), j, k, grid, args...)
    β4 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(3))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(VI.func, i + stencil_offset(bias, 4, 0), j, k, grid, args...)
    β5 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(4))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(VI.func, i + stencil_offset(bias, 5, 0), j, k, grid, args...)
    β6 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(5))

    # Compute weights from β values
    τ = global_smoothness_indicator(Val(6), (β1, β2, β3, β4, β5, β6))
    α = zweno_alpha_loop(scheme, (β1, β2, β3, β4, β5, β6), τ)
    Σα = α[1] + α[2] + α[3] + α[4] + α[5] + α[6]

    # Now compute all p values (reusing same registers)
    w1 = getvalue(ψ, i + stencil_offset(bias, 0, 0), j, k, grid, args...)
    w2 = getvalue(ψ, i + stencil_offset(bias, 0, 1), j, k, grid, args...)
    w3 = getvalue(ψ, i + stencil_offset(bias, 0, 2), j, k, grid, args...)
    w4 = getvalue(ψ, i + stencil_offset(bias, 0, 3), j, k, grid, args...)
    w5 = getvalue(ψ, i + stencil_offset(bias, 0, 4), j, k, grid, args...)
    w6 = getvalue(ψ, i + stencil_offset(bias, 0, 5), j, k, grid, args...)
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 1, 0), j, k, grid, args...)
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 2, 0), j, k, grid, args...)
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 3, 0), j, k, grid, args...)
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 4, 0), j, k, grid, args...)
    p5 = biased_p(scheme, bias, Val(4), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 5, 0), j, k, grid, args...)
    p6 = biased_p(scheme, bias, Val(5), (w1, w2, w3, w4, w5, w6))

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3 + α[4] * p4 + α[5] * p5 + α[6] * p6) / Σα
end

#####
##### Fused interpolation for VelocityStencil - y direction
#####

@inline function fused_biased_interpolate_yᵃᶠᵃ(i, j, k, grid,
                                               scheme::WENO{2, FT}, bias,
                                               ψ, ::VelocityStencil, u, v, args...) where FT
    # Compute all β values first (reusing registers for u and v)
    w1 = ℑyᵃᶠᵃ(i, j + stencil_offset(bias, 0, 0), k, grid, u)
    w2 = ℑyᵃᶠᵃ(i, j + stencil_offset(bias, 0, 1), k, grid, u)
    βu1 = smoothness_indicator((w1, w2), scheme, Val(0))

    w2 = w1
    w1 = ℑyᵃᶠᵃ(i, j + stencil_offset(bias, 1, 0), k, grid, u)
    βu2 = smoothness_indicator((w1, w2), scheme, Val(1))

    # Reuse same registers for v
    w1 = ℑxᶠᵃᵃ(i, j + stencil_offset(bias, 0, 0), k, grid, v)
    w2 = ℑxᶠᵃᵃ(i, j + stencil_offset(bias, 0, 1), k, grid, v)
    βv1 = smoothness_indicator((w1, w2), scheme, Val(0))

    w2 = w1
    w1 = ℑxᶠᵃᵃ(i, j + stencil_offset(bias, 1, 0), k, grid, v)
    βv2 = smoothness_indicator((w1, w2), scheme, Val(1))

    β1 = (βu1 + βv1) / 2
    β2 = (βu2 + βv2) / 2

    # Compute weights from β values
    τ = global_smoothness_indicator(Val(2), (β1, β2))
    α = zweno_alpha_loop(scheme, (β1, β2), τ)
    Σα = α[1] + α[2]

    # Now compute all p values (reusing same registers for w)
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 0, 0), k, grid, u, v, args...)
    w2 = getvalue(ψ, i, j + stencil_offset(bias, 0, 1), k, grid, u, v, args...)
    p1 = biased_p(scheme, bias, Val(0), (w1, w2))

    w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 1, 0), k, grid, u, v, args...)
    p2 = biased_p(scheme, bias, Val(1), (w1, w2))

    return @muladd (α[1] * p1 + α[2] * p2) / Σα
end

#####
##### Fused interpolation for FunctionStencil - y direction
#####

@inline function fused_biased_interpolate_yᵃᶠᵃ(i, j, k, grid,
                                               scheme::WENO{2, FT}, bias,
                                               ψ, VI::FunctionStencil, args...) where FT
    # Compute all β values first (reusing registers)
    w1 = getvalue(VI.func, i, j + stencil_offset(bias, 0, 0), k, grid, args...)
    w2 = getvalue(VI.func, i, j + stencil_offset(bias, 0, 1), k, grid, args...)
    β1 = smoothness_indicator((w1, w2), scheme, Val(0))

    w2 = w1
    w1 = getvalue(VI.func, i, j + stencil_offset(bias, 1, 0), k, grid, args...)
    β2 = smoothness_indicator((w1, w2), scheme, Val(1))

    # Compute weights from β values
    τ = global_smoothness_indicator(Val(2), (β1, β2))
    α = zweno_alpha_loop(scheme, (β1, β2), τ)
    Σα = α[1] + α[2]

    # Now compute all p values (reusing same registers)
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 0, 0), k, grid, args...)
    w2 = getvalue(ψ, i, j + stencil_offset(bias, 0, 1), k, grid, args...)
    p1 = biased_p(scheme, bias, Val(0), (w1, w2))

    w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 1, 0), k, grid, args...)
    p2 = biased_p(scheme, bias, Val(1), (w1, w2))

    return @muladd (α[1] * p1 + α[2] * p2) / Σα
end

#####
##### Fused interpolation for FunctionStencil - z direction
#####

@inline function fused_biased_interpolate_zᵃᵃᶠ(i, j, k, grid,
                                               scheme::WENO{2, FT}, bias,
                                               ψ, VI::FunctionStencil, args...) where FT
    # Compute all β values first (reusing registers)
    w1 = getvalue(VI.func, i, j, k + stencil_offset(bias, 0, 0), grid, args...)
    w2 = getvalue(VI.func, i, j, k + stencil_offset(bias, 0, 1), grid, args...)
    β1 = smoothness_indicator((w1, w2), scheme, Val(0))

    w2 = w1
    w1 = getvalue(VI.func, i, j, k + stencil_offset(bias, 1, 0), grid, args...)
    β2 = smoothness_indicator((w1, w2), scheme, Val(1))

    # Compute weights from β values
    τ = global_smoothness_indicator(Val(2), (β1, β2))
    α = zweno_alpha_loop(scheme, (β1, β2), τ)
    Σα = α[1] + α[2]

    # Now compute all p values (reusing same registers)
    w1 = getvalue(ψ, i, j, k + stencil_offset(bias, 0, 0), grid, args...)
    w2 = getvalue(ψ, i, j, k + stencil_offset(bias, 0, 1), grid, args...)
    p1 = biased_p(scheme, bias, Val(0), (w1, w2))

    w2 = w1
    w1 = getvalue(ψ, i, j, k + stencil_offset(bias, 1, 0), grid, args...)
    p2 = biased_p(scheme, bias, Val(1), (w1, w2))

    return @muladd (α[1] * p1 + α[2] * p2) / Σα
end

#####
##### VelocityStencil - y direction (continued)
#####

@inline function fused_biased_interpolate_yᵃᶠᵃ(i, j, k, grid,
                                               scheme::WENO{3, FT}, bias,
                                               ψ, ::VelocityStencil, u, v, args...) where FT
    # Compute all β values first (reusing registers for u and v)
    w1 = ℑyᵃᶠᵃ(i, j + stencil_offset(bias, 0, 0), k, grid, u)
    w2 = ℑyᵃᶠᵃ(i, j + stencil_offset(bias, 0, 1), k, grid, u)
    w3 = ℑyᵃᶠᵃ(i, j + stencil_offset(bias, 0, 2), k, grid, u)
    βu1 = smoothness_indicator((w1, w2, w3), scheme, Val(0))

    w3 = w2; w2 = w1
    w1 = ℑyᵃᶠᵃ(i, j + stencil_offset(bias, 1, 0), k, grid, u)
    βu2 = smoothness_indicator((w1, w2, w3), scheme, Val(1))

    w3 = w2; w2 = w1
    w1 = ℑyᵃᶠᵃ(i, j + stencil_offset(bias, 2, 0), k, grid, u)
    βu3 = smoothness_indicator((w1, w2, w3), scheme, Val(2))

    # Reuse same registers for v
    w1 = ℑxᶠᵃᵃ(i, j + stencil_offset(bias, 0, 0), k, grid, v)
    w2 = ℑxᶠᵃᵃ(i, j + stencil_offset(bias, 0, 1), k, grid, v)
    w3 = ℑxᶠᵃᵃ(i, j + stencil_offset(bias, 0, 2), k, grid, v)
    βv1 = smoothness_indicator((w1, w2, w3), scheme, Val(0))

    w3 = w2; w2 = w1
    w1 = ℑxᶠᵃᵃ(i, j + stencil_offset(bias, 1, 0), k, grid, v)
    βv2 = smoothness_indicator((w1, w2, w3), scheme, Val(1))

    w3 = w2; w2 = w1
    w1 = ℑxᶠᵃᵃ(i, j + stencil_offset(bias, 2, 0), k, grid, v)
    βv3 = smoothness_indicator((w1, w2, w3), scheme, Val(2))

    β1 = (βu1 + βv1) / 2
    β2 = (βu2 + βv2) / 2
    β3 = (βu3 + βv3) / 2

    # Compute weights from β values
    τ = global_smoothness_indicator(Val(3), (β1, β2, β3))
    α = zweno_alpha_loop(scheme, (β1, β2, β3), τ)
    Σα = α[1] + α[2] + α[3]

    # Now compute all p values (reusing same registers for w)
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 0, 0), k, grid, u, v, args...)
    w2 = getvalue(ψ, i, j + stencil_offset(bias, 0, 1), k, grid, u, v, args...)
    w3 = getvalue(ψ, i, j + stencil_offset(bias, 0, 2), k, grid, u, v, args...)
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3))

    w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 1, 0), k, grid, u, v, args...)
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3))

    w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 2, 0), k, grid, u, v, args...)
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3))

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3) / Σα
end

@inline function fused_biased_interpolate_yᵃᶠᵃ(i, j, k, grid,
                                               scheme::WENO{4, FT}, bias,
                                               ψ, ::VelocityStencil, u, v, args...) where FT
    # Compute all β values first (reusing registers for u and v)
    w1 = ℑyᵃᶠᵃ(i, j + stencil_offset(bias, 0, 0), k, grid, u)
    w2 = ℑyᵃᶠᵃ(i, j + stencil_offset(bias, 0, 1), k, grid, u)
    w3 = ℑyᵃᶠᵃ(i, j + stencil_offset(bias, 0, 2), k, grid, u)
    w4 = ℑyᵃᶠᵃ(i, j + stencil_offset(bias, 0, 3), k, grid, u)
    βu1 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(0))

    w4 = w3; w3 = w2; w2 = w1
    w1 = ℑyᵃᶠᵃ(i, j + stencil_offset(bias, 1, 0), k, grid, u)
    βu2 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(1))

    w4 = w3; w3 = w2; w2 = w1
    w1 = ℑyᵃᶠᵃ(i, j + stencil_offset(bias, 2, 0), k, grid, u)
    βu3 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(2))

    w4 = w3; w3 = w2; w2 = w1
    w1 = ℑyᵃᶠᵃ(i, j + stencil_offset(bias, 3, 0), k, grid, u)
    βu4 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(3))

    # Reuse same registers for v
    w1 = ℑxᶠᵃᵃ(i, j + stencil_offset(bias, 0, 0), k, grid, v)
    w2 = ℑxᶠᵃᵃ(i, j + stencil_offset(bias, 0, 1), k, grid, v)
    w3 = ℑxᶠᵃᵃ(i, j + stencil_offset(bias, 0, 2), k, grid, v)
    w4 = ℑxᶠᵃᵃ(i, j + stencil_offset(bias, 0, 3), k, grid, v)
    βv1 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(0))

    w4 = w3; w3 = w2; w2 = w1
    w1 = ℑxᶠᵃᵃ(i, j + stencil_offset(bias, 1, 0), k, grid, v)
    βv2 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(1))

    w4 = w3; w3 = w2; w2 = w1
    w1 = ℑxᶠᵃᵃ(i, j + stencil_offset(bias, 2, 0), k, grid, v)
    βv3 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(2))

    w4 = w3; w3 = w2; w2 = w1
    w1 = ℑxᶠᵃᵃ(i, j + stencil_offset(bias, 3, 0), k, grid, v)
    βv4 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(3))

    β1 = (βu1 + βv1) / 2
    β2 = (βu2 + βv2) / 2
    β3 = (βu3 + βv3) / 2
    β4 = (βu4 + βv4) / 2

    # Compute weights from β values
    τ = global_smoothness_indicator(Val(4), (β1, β2, β3, β4))
    α = zweno_alpha_loop(scheme, (β1, β2, β3, β4), τ)
    Σα = α[1] + α[2] + α[3] + α[4]

    # Now compute all p values (reusing same registers for w)
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 0, 0), k, grid, u, v, args...)
    w2 = getvalue(ψ, i, j + stencil_offset(bias, 0, 1), k, grid, u, v, args...)
    w3 = getvalue(ψ, i, j + stencil_offset(bias, 0, 2), k, grid, u, v, args...)
    w4 = getvalue(ψ, i, j + stencil_offset(bias, 0, 3), k, grid, u, v, args...)
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4))

    w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 1, 0), k, grid, u, v, args...)
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4))

    w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 2, 0), k, grid, u, v, args...)
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4))

    w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 3, 0), k, grid, u, v, args...)
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4))

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3 + α[4] * p4) / Σα
end

@inline function fused_biased_interpolate_yᵃᶠᵃ(i, j, k, grid,
                                               scheme::WENO{5, FT}, bias,
                                               ψ, ::VelocityStencil, u, v, args...) where FT
    # Compute all β values first (reusing registers for u and v)
    w1 = ℑyᵃᶠᵃ(i, j + stencil_offset(bias, 0, 0), k, grid, u)
    w2 = ℑyᵃᶠᵃ(i, j + stencil_offset(bias, 0, 1), k, grid, u)
    w3 = ℑyᵃᶠᵃ(i, j + stencil_offset(bias, 0, 2), k, grid, u)
    w4 = ℑyᵃᶠᵃ(i, j + stencil_offset(bias, 0, 3), k, grid, u)
    w5 = ℑyᵃᶠᵃ(i, j + stencil_offset(bias, 0, 4), k, grid, u)
    βu1 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(0))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ℑyᵃᶠᵃ(i, j + stencil_offset(bias, 1, 0), k, grid, u)
    βu2 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(1))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ℑyᵃᶠᵃ(i, j + stencil_offset(bias, 2, 0), k, grid, u)
    βu3 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(2))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ℑyᵃᶠᵃ(i, j + stencil_offset(bias, 3, 0), k, grid, u)
    βu4 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(3))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ℑyᵃᶠᵃ(i, j + stencil_offset(bias, 4, 0), k, grid, u)
    βu5 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(4))

    # Reuse same registers for v
    w1 = ℑxᶠᵃᵃ(i, j + stencil_offset(bias, 0, 0), k, grid, v)
    w2 = ℑxᶠᵃᵃ(i, j + stencil_offset(bias, 0, 1), k, grid, v)
    w3 = ℑxᶠᵃᵃ(i, j + stencil_offset(bias, 0, 2), k, grid, v)
    w4 = ℑxᶠᵃᵃ(i, j + stencil_offset(bias, 0, 3), k, grid, v)
    w5 = ℑxᶠᵃᵃ(i, j + stencil_offset(bias, 0, 4), k, grid, v)
    βv1 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(0))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ℑxᶠᵃᵃ(i, j + stencil_offset(bias, 1, 0), k, grid, v)
    βv2 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(1))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ℑxᶠᵃᵃ(i, j + stencil_offset(bias, 2, 0), k, grid, v)
    βv3 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(2))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ℑxᶠᵃᵃ(i, j + stencil_offset(bias, 3, 0), k, grid, v)
    βv4 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(3))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ℑxᶠᵃᵃ(i, j + stencil_offset(bias, 4, 0), k, grid, v)
    βv5 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(4))

    β1 = (βu1 + βv1) / 2
    β2 = (βu2 + βv2) / 2
    β3 = (βu3 + βv3) / 2
    β4 = (βu4 + βv4) / 2
    β5 = (βu5 + βv5) / 2

    # Compute weights from β values
    τ = global_smoothness_indicator(Val(5), (β1, β2, β3, β4, β5))
    α = zweno_alpha_loop(scheme, (β1, β2, β3, β4, β5), τ)
    Σα = α[1] + α[2] + α[3] + α[4] + α[5]

    # Now compute all p values (reusing same registers for w)
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 0, 0), k, grid, u, v, args...)
    w2 = getvalue(ψ, i, j + stencil_offset(bias, 0, 1), k, grid, u, v, args...)
    w3 = getvalue(ψ, i, j + stencil_offset(bias, 0, 2), k, grid, u, v, args...)
    w4 = getvalue(ψ, i, j + stencil_offset(bias, 0, 3), k, grid, u, v, args...)
    w5 = getvalue(ψ, i, j + stencil_offset(bias, 0, 4), k, grid, u, v, args...)
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 1, 0), k, grid, u, v, args...)
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 2, 0), k, grid, u, v, args...)
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 3, 0), k, grid, u, v, args...)
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 4, 0), k, grid, u, v, args...)
    p5 = biased_p(scheme, bias, Val(4), (w1, w2, w3, w4, w5))

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3 + α[4] * p4 + α[5] * p5) / Σα
end

@inline function fused_biased_interpolate_yᵃᶠᵃ(i, j, k, grid,
                                               scheme::WENO{6, FT}, bias,
                                               ψ, ::VelocityStencil, u, v, args...) where FT
    # Compute all β values first (reusing registers for u and v)
    w1 = ℑyᵃᶠᵃ(i, j + stencil_offset(bias, 0, 0), k, grid, u)
    w2 = ℑyᵃᶠᵃ(i, j + stencil_offset(bias, 0, 1), k, grid, u)
    w3 = ℑyᵃᶠᵃ(i, j + stencil_offset(bias, 0, 2), k, grid, u)
    w4 = ℑyᵃᶠᵃ(i, j + stencil_offset(bias, 0, 3), k, grid, u)
    w5 = ℑyᵃᶠᵃ(i, j + stencil_offset(bias, 0, 4), k, grid, u)
    w6 = ℑyᵃᶠᵃ(i, j + stencil_offset(bias, 0, 5), k, grid, u)
    βu1 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(0))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ℑyᵃᶠᵃ(i, j + stencil_offset(bias, 1, 0), k, grid, u)
    βu2 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(1))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ℑyᵃᶠᵃ(i, j + stencil_offset(bias, 2, 0), k, grid, u)
    βu3 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(2))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ℑyᵃᶠᵃ(i, j + stencil_offset(bias, 3, 0), k, grid, u)
    βu4 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(3))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ℑyᵃᶠᵃ(i, j + stencil_offset(bias, 4, 0), k, grid, u)
    βu5 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(4))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ℑyᵃᶠᵃ(i, j + stencil_offset(bias, 5, 0), k, grid, u)
    βu6 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(5))

    # Reuse same registers for v
    w1 = ℑxᶠᵃᵃ(i, j + stencil_offset(bias, 0, 0), k, grid, v)
    w2 = ℑxᶠᵃᵃ(i, j + stencil_offset(bias, 0, 1), k, grid, v)
    w3 = ℑxᶠᵃᵃ(i, j + stencil_offset(bias, 0, 2), k, grid, v)
    w4 = ℑxᶠᵃᵃ(i, j + stencil_offset(bias, 0, 3), k, grid, v)
    w5 = ℑxᶠᵃᵃ(i, j + stencil_offset(bias, 0, 4), k, grid, v)
    w6 = ℑxᶠᵃᵃ(i, j + stencil_offset(bias, 0, 5), k, grid, v)
    βv1 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(0))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ℑxᶠᵃᵃ(i, j + stencil_offset(bias, 1, 0), k, grid, v)
    βv2 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(1))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ℑxᶠᵃᵃ(i, j + stencil_offset(bias, 2, 0), k, grid, v)
    βv3 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(2))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ℑxᶠᵃᵃ(i, j + stencil_offset(bias, 3, 0), k, grid, v)
    βv4 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(3))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ℑxᶠᵃᵃ(i, j + stencil_offset(bias, 4, 0), k, grid, v)
    βv5 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(4))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ℑxᶠᵃᵃ(i, j + stencil_offset(bias, 5, 0), k, grid, v)
    βv6 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(5))

    β1 = (βu1 + βv1) / 2
    β2 = (βu2 + βv2) / 2
    β3 = (βu3 + βv3) / 2
    β4 = (βu4 + βv4) / 2
    β5 = (βu5 + βv5) / 2
    β6 = (βu6 + βv6) / 2

    # Compute weights from β values
    τ = global_smoothness_indicator(Val(6), (β1, β2, β3, β4, β5, β6))
    α = zweno_alpha_loop(scheme, (β1, β2, β3, β4, β5, β6), τ)
    Σα = α[1] + α[2] + α[3] + α[4] + α[5] + α[6]

    # Now compute all p values (reusing same registers for w)
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 0, 0), k, grid, u, v, args...)
    w2 = getvalue(ψ, i, j + stencil_offset(bias, 0, 1), k, grid, u, v, args...)
    w3 = getvalue(ψ, i, j + stencil_offset(bias, 0, 2), k, grid, u, v, args...)
    w4 = getvalue(ψ, i, j + stencil_offset(bias, 0, 3), k, grid, u, v, args...)
    w5 = getvalue(ψ, i, j + stencil_offset(bias, 0, 4), k, grid, u, v, args...)
    w6 = getvalue(ψ, i, j + stencil_offset(bias, 0, 5), k, grid, u, v, args...)
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 1, 0), k, grid, u, v, args...)
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 2, 0), k, grid, u, v, args...)
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 3, 0), k, grid, u, v, args...)
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 4, 0), k, grid, u, v, args...)
    p5 = biased_p(scheme, bias, Val(4), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 5, 0), k, grid, u, v, args...)
    p6 = biased_p(scheme, bias, Val(5), (w1, w2, w3, w4, w5, w6))

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3 + α[4] * p4 + α[5] * p5 + α[6] * p6) / Σα
end

#####
##### FunctionStencil - y direction (continued)
#####

@inline function fused_biased_interpolate_yᵃᶠᵃ(i, j, k, grid,
                                               scheme::WENO{3, FT}, bias,
                                               ψ, VI::FunctionStencil, args...) where FT
    # Compute all β values first (reusing registers)
    w1 = getvalue(VI.func, i, j + stencil_offset(bias, 0, 0), k, grid, args...)
    w2 = getvalue(VI.func, i, j + stencil_offset(bias, 0, 1), k, grid, args...)
    w3 = getvalue(VI.func, i, j + stencil_offset(bias, 0, 2), k, grid, args...)
    β1 = smoothness_indicator((w1, w2, w3), scheme, Val(0))

    w3 = w2; w2 = w1
    w1 = getvalue(VI.func, i, j + stencil_offset(bias, 1, 0), k, grid, args...)
    β2 = smoothness_indicator((w1, w2, w3), scheme, Val(1))

    w3 = w2; w2 = w1
    w1 = getvalue(VI.func, i, j + stencil_offset(bias, 2, 0), k, grid, args...)
    β3 = smoothness_indicator((w1, w2, w3), scheme, Val(2))

    # Compute weights from β values
    τ = global_smoothness_indicator(Val(3), (β1, β2, β3))
    α = zweno_alpha_loop(scheme, (β1, β2, β3), τ)
    Σα = α[1] + α[2] + α[3]

    # Now compute all p values (reusing same registers)
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 0, 0), k, grid, args...)
    w2 = getvalue(ψ, i, j + stencil_offset(bias, 0, 1), k, grid, args...)
    w3 = getvalue(ψ, i, j + stencil_offset(bias, 0, 2), k, grid, args...)
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3))

    w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 1, 0), k, grid, args...)
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3))

    w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 2, 0), k, grid, args...)
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3))

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3) / Σα
end

@inline function fused_biased_interpolate_yᵃᶠᵃ(i, j, k, grid,
                                               scheme::WENO{4, FT}, bias,
                                               ψ, VI::FunctionStencil, args...) where FT
    # Compute all β values first (reusing registers)
    w1 = getvalue(VI.func, i, j + stencil_offset(bias, 0, 0), k, grid, args...)
    w2 = getvalue(VI.func, i, j + stencil_offset(bias, 0, 1), k, grid, args...)
    w3 = getvalue(VI.func, i, j + stencil_offset(bias, 0, 2), k, grid, args...)
    w4 = getvalue(VI.func, i, j + stencil_offset(bias, 0, 3), k, grid, args...)
    β1 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(0))

    w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(VI.func, i, j + stencil_offset(bias, 1, 0), k, grid, args...)
    β2 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(1))

    w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(VI.func, i, j + stencil_offset(bias, 2, 0), k, grid, args...)
    β3 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(2))

    w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(VI.func, i, j + stencil_offset(bias, 3, 0), k, grid, args...)
    β4 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(3))

    # Compute weights from β values
    τ = global_smoothness_indicator(Val(4), (β1, β2, β3, β4))
    α = zweno_alpha_loop(scheme, (β1, β2, β3, β4), τ)
    Σα = α[1] + α[2] + α[3] + α[4]

    # Now compute all p values (reusing same registers)
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 0, 0), k, grid, args...)
    w2 = getvalue(ψ, i, j + stencil_offset(bias, 0, 1), k, grid, args...)
    w3 = getvalue(ψ, i, j + stencil_offset(bias, 0, 2), k, grid, args...)
    w4 = getvalue(ψ, i, j + stencil_offset(bias, 0, 3), k, grid, args...)
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4))

    w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 1, 0), k, grid, args...)
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4))

    w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 2, 0), k, grid, args...)
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4))

    w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 3, 0), k, grid, args...)
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4))

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3 + α[4] * p4) / Σα
end

@inline function fused_biased_interpolate_yᵃᶠᵃ(i, j, k, grid,
                                               scheme::WENO{5, FT}, bias,
                                               ψ, VI::FunctionStencil, args...) where FT
    # Compute all β values first (reusing registers)
    w1 = getvalue(VI.func, i, j + stencil_offset(bias, 0, 0), k, grid, args...)
    w2 = getvalue(VI.func, i, j + stencil_offset(bias, 0, 1), k, grid, args...)
    w3 = getvalue(VI.func, i, j + stencil_offset(bias, 0, 2), k, grid, args...)
    w4 = getvalue(VI.func, i, j + stencil_offset(bias, 0, 3), k, grid, args...)
    w5 = getvalue(VI.func, i, j + stencil_offset(bias, 0, 4), k, grid, args...)
    β1 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(0))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(VI.func, i, j + stencil_offset(bias, 1, 0), k, grid, args...)
    β2 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(1))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(VI.func, i, j + stencil_offset(bias, 2, 0), k, grid, args...)
    β3 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(2))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(VI.func, i, j + stencil_offset(bias, 3, 0), k, grid, args...)
    β4 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(3))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(VI.func, i, j + stencil_offset(bias, 4, 0), k, grid, args...)
    β5 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(4))

    # Compute weights from β values
    τ = global_smoothness_indicator(Val(5), (β1, β2, β3, β4, β5))
    α = zweno_alpha_loop(scheme, (β1, β2, β3, β4, β5), τ)
    Σα = α[1] + α[2] + α[3] + α[4] + α[5]

    # Now compute all p values (reusing same registers)
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 0, 0), k, grid, args...)
    w2 = getvalue(ψ, i, j + stencil_offset(bias, 0, 1), k, grid, args...)
    w3 = getvalue(ψ, i, j + stencil_offset(bias, 0, 2), k, grid, args...)
    w4 = getvalue(ψ, i, j + stencil_offset(bias, 0, 3), k, grid, args...)
    w5 = getvalue(ψ, i, j + stencil_offset(bias, 0, 4), k, grid, args...)
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 1, 0), k, grid, args...)
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 2, 0), k, grid, args...)
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 3, 0), k, grid, args...)
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 4, 0), k, grid, args...)
    p5 = biased_p(scheme, bias, Val(4), (w1, w2, w3, w4, w5))

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3 + α[4] * p4 + α[5] * p5) / Σα
end

@inline function fused_biased_interpolate_yᵃᶠᵃ(i, j, k, grid,
                                               scheme::WENO{6, FT}, bias,
                                               ψ, VI::FunctionStencil, args...) where FT
    # Compute all β values first (reusing registers)
    w1 = getvalue(VI.func, i, j + stencil_offset(bias, 0, 0), k, grid, args...)
    w2 = getvalue(VI.func, i, j + stencil_offset(bias, 0, 1), k, grid, args...)
    w3 = getvalue(VI.func, i, j + stencil_offset(bias, 0, 2), k, grid, args...)
    w4 = getvalue(VI.func, i, j + stencil_offset(bias, 0, 3), k, grid, args...)
    w5 = getvalue(VI.func, i, j + stencil_offset(bias, 0, 4), k, grid, args...)
    w6 = getvalue(VI.func, i, j + stencil_offset(bias, 0, 5), k, grid, args...)
    β1 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(0))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(VI.func, i, j + stencil_offset(bias, 1, 0), k, grid, args...)
    β2 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(1))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(VI.func, i, j + stencil_offset(bias, 2, 0), k, grid, args...)
    β3 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(2))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(VI.func, i, j + stencil_offset(bias, 3, 0), k, grid, args...)
    β4 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(3))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(VI.func, i, j + stencil_offset(bias, 4, 0), k, grid, args...)
    β5 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(4))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(VI.func, i, j + stencil_offset(bias, 5, 0), k, grid, args...)
    β6 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(5))

    # Compute weights from β values
    τ = global_smoothness_indicator(Val(6), (β1, β2, β3, β4, β5, β6))
    α = zweno_alpha_loop(scheme, (β1, β2, β3, β4, β5, β6), τ)
    Σα = α[1] + α[2] + α[3] + α[4] + α[5] + α[6]

    # Now compute all p values (reusing same registers)
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 0, 0), k, grid, args...)
    w2 = getvalue(ψ, i, j + stencil_offset(bias, 0, 1), k, grid, args...)
    w3 = getvalue(ψ, i, j + stencil_offset(bias, 0, 2), k, grid, args...)
    w4 = getvalue(ψ, i, j + stencil_offset(bias, 0, 3), k, grid, args...)
    w5 = getvalue(ψ, i, j + stencil_offset(bias, 0, 4), k, grid, args...)
    w6 = getvalue(ψ, i, j + stencil_offset(bias, 0, 5), k, grid, args...)
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 1, 0), k, grid, args...)
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 2, 0), k, grid, args...)
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 3, 0), k, grid, args...)
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 4, 0), k, grid, args...)
    p5 = biased_p(scheme, bias, Val(4), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 5, 0), k, grid, args...)
    p6 = biased_p(scheme, bias, Val(5), (w1, w2, w3, w4, w5, w6))

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3 + α[4] * p4 + α[5] * p5 + α[6] * p6) / Σα
end

#####
##### FunctionStencil - z direction (continued)
#####

@inline function fused_biased_interpolate_zᵃᵃᶠ(i, j, k, grid,
                                               scheme::WENO{3, FT}, bias,
                                               ψ, VI::FunctionStencil, args...) where FT
    # Compute all β values first (reusing registers)
    w1 = getvalue(VI.func, i, j, k + stencil_offset(bias, 0, 0), grid, args...)
    w2 = getvalue(VI.func, i, j, k + stencil_offset(bias, 0, 1), grid, args...)
    w3 = getvalue(VI.func, i, j, k + stencil_offset(bias, 0, 2), grid, args...)
    β1 = smoothness_indicator((w1, w2, w3), scheme, Val(0))

    w3 = w2; w2 = w1
    w1 = getvalue(VI.func, i, j, k + stencil_offset(bias, 1, 0), grid, args...)
    β2 = smoothness_indicator((w1, w2, w3), scheme, Val(1))

    w3 = w2; w2 = w1
    w1 = getvalue(VI.func, i, j, k + stencil_offset(bias, 2, 0), grid, args...)
    β3 = smoothness_indicator((w1, w2, w3), scheme, Val(2))

    # Compute weights from β values
    τ = global_smoothness_indicator(Val(3), (β1, β2, β3))
    α = zweno_alpha_loop(scheme, (β1, β2, β3), τ)
    Σα = α[1] + α[2] + α[3]

    # Now compute all p values (reusing same registers)
    w1 = getvalue(ψ, i, j, k + stencil_offset(bias, 0, 0), grid, args...)
    w2 = getvalue(ψ, i, j, k + stencil_offset(bias, 0, 1), grid, args...)
    w3 = getvalue(ψ, i, j, k + stencil_offset(bias, 0, 2), grid, args...)
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3))

    w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j, k + stencil_offset(bias, 1, 0), grid, args...)
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3))

    w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j, k + stencil_offset(bias, 2, 0), grid, args...)
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3))

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3) / Σα
end

@inline function fused_biased_interpolate_zᵃᵃᶠ(i, j, k, grid,
                                               scheme::WENO{4, FT}, bias,
                                               ψ, VI::FunctionStencil, args...) where FT
    # Compute all β values first (reusing registers)
    w1 = getvalue(VI.func, i, j, k + stencil_offset(bias, 0, 0), grid, args...)
    w2 = getvalue(VI.func, i, j, k + stencil_offset(bias, 0, 1), grid, args...)
    w3 = getvalue(VI.func, i, j, k + stencil_offset(bias, 0, 2), grid, args...)
    w4 = getvalue(VI.func, i, j, k + stencil_offset(bias, 0, 3), grid, args...)
    β1 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(0))

    w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(VI.func, i, j, k + stencil_offset(bias, 1, 0), grid, args...)
    β2 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(1))

    w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(VI.func, i, j, k + stencil_offset(bias, 2, 0), grid, args...)
    β3 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(2))

    w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(VI.func, i, j, k + stencil_offset(bias, 3, 0), grid, args...)
    β4 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(3))

    # Compute weights from β values
    τ = global_smoothness_indicator(Val(4), (β1, β2, β3, β4))
    α = zweno_alpha_loop(scheme, (β1, β2, β3, β4), τ)
    Σα = α[1] + α[2] + α[3] + α[4]

    # Now compute all p values (reusing same registers)
    w1 = getvalue(ψ, i, j, k + stencil_offset(bias, 0, 0), grid, args...)
    w2 = getvalue(ψ, i, j, k + stencil_offset(bias, 0, 1), grid, args...)
    w3 = getvalue(ψ, i, j, k + stencil_offset(bias, 0, 2), grid, args...)
    w4 = getvalue(ψ, i, j, k + stencil_offset(bias, 0, 3), grid, args...)
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4))

    w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j, k + stencil_offset(bias, 1, 0), grid, args...)
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4))

    w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j, k + stencil_offset(bias, 2, 0), grid, args...)
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4))

    w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j, k + stencil_offset(bias, 3, 0), grid, args...)
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4))

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3 + α[4] * p4) / Σα
end

@inline function fused_biased_interpolate_zᵃᵃᶠ(i, j, k, grid,
                                               scheme::WENO{5, FT}, bias,
                                               ψ, VI::FunctionStencil, args...) where FT
    # Compute all β values first (reusing registers)
    w1 = getvalue(VI.func, i, j, k + stencil_offset(bias, 0, 0), grid, args...)
    w2 = getvalue(VI.func, i, j, k + stencil_offset(bias, 0, 1), grid, args...)
    w3 = getvalue(VI.func, i, j, k + stencil_offset(bias, 0, 2), grid, args...)
    w4 = getvalue(VI.func, i, j, k + stencil_offset(bias, 0, 3), grid, args...)
    w5 = getvalue(VI.func, i, j, k + stencil_offset(bias, 0, 4), grid, args...)
    β1 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(0))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(VI.func, i, j, k + stencil_offset(bias, 1, 0), grid, args...)
    β2 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(1))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(VI.func, i, j, k + stencil_offset(bias, 2, 0), grid, args...)
    β3 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(2))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(VI.func, i, j, k + stencil_offset(bias, 3, 0), grid, args...)
    β4 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(3))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(VI.func, i, j, k + stencil_offset(bias, 4, 0), grid, args...)
    β5 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(4))

    # Compute weights from β values
    τ = global_smoothness_indicator(Val(5), (β1, β2, β3, β4, β5))
    α = zweno_alpha_loop(scheme, (β1, β2, β3, β4, β5), τ)
    Σα = α[1] + α[2] + α[3] + α[4] + α[5]

    # Now compute all p values (reusing same registers)
    w1 = getvalue(ψ, i, j, k + stencil_offset(bias, 0, 0), grid, args...)
    w2 = getvalue(ψ, i, j, k + stencil_offset(bias, 0, 1), grid, args...)
    w3 = getvalue(ψ, i, j, k + stencil_offset(bias, 0, 2), grid, args...)
    w4 = getvalue(ψ, i, j, k + stencil_offset(bias, 0, 3), grid, args...)
    w5 = getvalue(ψ, i, j, k + stencil_offset(bias, 0, 4), grid, args...)
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j, k + stencil_offset(bias, 1, 0), grid, args...)
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j, k + stencil_offset(bias, 2, 0), grid, args...)
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j, k + stencil_offset(bias, 3, 0), grid, args...)
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j, k + stencil_offset(bias, 4, 0), grid, args...)
    p5 = biased_p(scheme, bias, Val(4), (w1, w2, w3, w4, w5))

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3 + α[4] * p4 + α[5] * p5) / Σα
end

@inline function fused_biased_interpolate_zᵃᵃᶠ(i, j, k, grid,
                                               scheme::WENO{6, FT}, bias,
                                               ψ, VI::FunctionStencil, args...) where FT
    # Compute all β values first (reusing registers)
    w1 = getvalue(VI.func, i, j, k + stencil_offset(bias, 0, 0), grid, args...)
    w2 = getvalue(VI.func, i, j, k + stencil_offset(bias, 0, 1), grid, args...)
    w3 = getvalue(VI.func, i, j, k + stencil_offset(bias, 0, 2), grid, args...)
    w4 = getvalue(VI.func, i, j, k + stencil_offset(bias, 0, 3), grid, args...)
    w5 = getvalue(VI.func, i, j, k + stencil_offset(bias, 0, 4), grid, args...)
    w6 = getvalue(VI.func, i, j, k + stencil_offset(bias, 0, 5), grid, args...)
    β1 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(0))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(VI.func, i, j, k + stencil_offset(bias, 1, 0), grid, args...)
    β2 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(1))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(VI.func, i, j, k + stencil_offset(bias, 2, 0), grid, args...)
    β3 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(2))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(VI.func, i, j, k + stencil_offset(bias, 3, 0), grid, args...)
    β4 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(3))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(VI.func, i, j, k + stencil_offset(bias, 4, 0), grid, args...)
    β5 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(4))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(VI.func, i, j, k + stencil_offset(bias, 5, 0), grid, args...)
    β6 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(5))

    # Compute weights from β values
    τ = global_smoothness_indicator(Val(6), (β1, β2, β3, β4, β5, β6))
    α = zweno_alpha_loop(scheme, (β1, β2, β3, β4, β5, β6), τ)
    Σα = α[1] + α[2] + α[3] + α[4] + α[5] + α[6]

    # Now compute all p values (reusing same registers)
    w1 = getvalue(ψ, i, j, k + stencil_offset(bias, 0, 0), grid, args...)
    w2 = getvalue(ψ, i, j, k + stencil_offset(bias, 0, 1), grid, args...)
    w3 = getvalue(ψ, i, j, k + stencil_offset(bias, 0, 2), grid, args...)
    w4 = getvalue(ψ, i, j, k + stencil_offset(bias, 0, 3), grid, args...)
    w5 = getvalue(ψ, i, j, k + stencil_offset(bias, 0, 4), grid, args...)
    w6 = getvalue(ψ, i, j, k + stencil_offset(bias, 0, 5), grid, args...)
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j, k + stencil_offset(bias, 1, 0), grid, args...)
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j, k + stencil_offset(bias, 2, 0), grid, args...)
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j, k + stencil_offset(bias, 3, 0), grid, args...)
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j, k + stencil_offset(bias, 4, 0), grid, args...)
    p5 = biased_p(scheme, bias, Val(4), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j, k + stencil_offset(bias, 5, 0), grid, args...)
    p6 = biased_p(scheme, bias, Val(5), (w1, w2, w3, w4, w5, w6))

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3 + α[4] * p4 + α[5] * p5 + α[6] * p6) / Σα
end
