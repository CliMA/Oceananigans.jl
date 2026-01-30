#####
##### Low-register WENO interpolation using sliding window approach
#####
##### Instead of loading all stencils at once (high register pressure), we:
##### 1. Load initial window of `buffer` values
##### 2. For each stencil: compute β and biased_p, then slide window by one value
##### 3. Compute weights from all β values
##### 4. Return weighted sum of biased_p values
#####
##### This reduces register usage from O(buffer²) to O(buffer)
#####

#####
##### Helper function to unify array and function access
#####

@inline getvalue(a, i, j, k, grid, args...) = @inbounds a[i, j, k]
@inline getvalue(a::Function, i, j, k, grid, args...) = a(i, j, k, grid, args...)

#####
##### Helper function to compute index offset based on bias, stencil, and position
#####

@inline stencil_offset(::LeftBias, s, p)  = p - s - 1
@inline stencil_offset(::RightBias, s, p) = s - p

#####
##### WENO{2} (3rd order) - x direction
#####

@inline function fused_biased_interpolate_xᶠᵃᵃ(i, j, k, grid,
                                               scheme::WENO{2, FT}, bias,
                                               ψ, args...) where FT
    # Stencil 0: load initial window
    w1 = getvalue(ψ, i + stencil_offset(bias, 0, 0), j, k, grid, args...)
    w2 = getvalue(ψ, i + stencil_offset(bias, 0, 1), j, k, grid, args...)

    β1 = smoothness_indicator((w1, w2), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2))

    # Slide for stencil 1
    w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 1, 0), j, k, grid, args...)

    β2 = smoothness_indicator((w1, w2), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2))

    # Compute weights
    τ = global_smoothness_indicator(Val(2), (β1, β2))
    α = zweno_alpha_loop(scheme, (β1, β2), τ)
    Σα = α[1] + α[2]

    return @muladd (α[1] * p1 + α[2] * p2) / Σα
end

#####
##### WENO{2} (3rd order) - y direction
#####

@inline function fused_biased_interpolate_yᵃᶠᵃ(i, j, k, grid,
                                                 scheme::WENO{2, FT}, bias,
                                                 ψ, args...) where FT
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 0, 0), k, grid, args...)
    w2 = getvalue(ψ, i, j + stencil_offset(bias, 0, 1), k, grid, args...)

    β1 = smoothness_indicator((w1, w2), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2))

    w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 1, 0), k, grid, args...)

    β2 = smoothness_indicator((w1, w2), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2))

    τ = global_smoothness_indicator(Val(2), (β1, β2))
    α = zweno_alpha_loop(scheme, (β1, β2), τ)
    Σα = α[1] + α[2]

    return @muladd (α[1] * p1 + α[2] * p2) / Σα
end

#####
##### WENO{2} (3rd order) - z direction
#####

@inline function fused_biased_interpolate_zᵃᵃᶠ(i, j, k, grid,
                                                 scheme::WENO{2, FT}, bias,
                                                 ψ, args...) where FT
    w1 = getvalue(ψ, i, j, k + stencil_offset(bias, 0, 0), grid, args...)
    w2 = getvalue(ψ, i, j, k + stencil_offset(bias, 0, 1), grid, args...)

    β1 = smoothness_indicator((w1, w2), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2))

    w2 = w1
    w1 = getvalue(ψ, i, j, k + stencil_offset(bias, 1, 0), grid, args...)

    β2 = smoothness_indicator((w1, w2), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2))

    τ = global_smoothness_indicator(Val(2), (β1, β2))
    α = zweno_alpha_loop(scheme, (β1, β2), τ)
    Σα = α[1] + α[2]

    return @muladd (α[1] * p1 + α[2] * p2) / Σα
end

#####
##### WENO{3} (5th order) - x direction
#####

@inline function fused_biased_interpolate_xᶠᵃᵃ(i, j, k, grid,
                                                 scheme::WENO{3, FT}, bias,
                                                 ψ, args...) where FT
    # Stencil 0: load initial window
    w1 = getvalue(ψ, i + stencil_offset(bias, 0, 0), j, k, grid, args...)
    w2 = getvalue(ψ, i + stencil_offset(bias, 0, 1), j, k, grid, args...)
    w3 = getvalue(ψ, i + stencil_offset(bias, 0, 2), j, k, grid, args...)

    β1 = smoothness_indicator((w1, w2, w3), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3))

    # Slide for stencil 1
    w3 = w2
    w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 1, 0), j, k, grid, args...)

    β2 = smoothness_indicator((w1, w2, w3), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3))

    # Slide for stencil 2
    w3 = w2
    w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 2, 0), j, k, grid, args...)

    β3 = smoothness_indicator((w1, w2, w3), scheme, Val(2))
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3))

    # Compute weights
    τ = global_smoothness_indicator(Val(3), (β1, β2, β3))
    α = zweno_alpha_loop(scheme, (β1, β2, β3), τ)
    Σα = α[1] + α[2] + α[3]

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3) / Σα
end

#####
##### WENO{3} (5th order) - y direction
#####

@inline function fused_biased_interpolate_yᵃᶠᵃ(i, j, k, grid,
                                                 scheme::WENO{3, FT}, bias,
                                                 ψ, args...) where FT
    # Stencil 0: load initial window
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 0, 0), k, grid, args...)
    w2 = getvalue(ψ, i, j + stencil_offset(bias, 0, 1), k, grid, args...)
    w3 = getvalue(ψ, i, j + stencil_offset(bias, 0, 2), k, grid, args...)

    β1 = smoothness_indicator((w1, w2, w3), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3))

    # Slide for stencil 1
    w3 = w2
    w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 1, 0), k, grid, args...)

    β2 = smoothness_indicator((w1, w2, w3), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3))

    # Slide for stencil 2
    w3 = w2
    w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 2, 0), k, grid, args...)

    β3 = smoothness_indicator((w1, w2, w3), scheme, Val(2))
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3))

    τ = global_smoothness_indicator(Val(3), (β1, β2, β3))
    α = zweno_alpha_loop(scheme, (β1, β2, β3), τ)
    Σα = α[1] + α[2] + α[3]

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3) / Σα
end

#####
##### WENO{3} (5th order) - z direction
#####

@inline function fused_biased_interpolate_zᵃᵃᶠ(i, j, k, grid,
                                                 scheme::WENO{3, FT}, bias,
                                                 ψ, args...) where FT
    # Stencil 0: load initial window
    w1 = getvalue(ψ, i, j, k + stencil_offset(bias, 0, 0), grid, args...)
    w2 = getvalue(ψ, i, j, k + stencil_offset(bias, 0, 1), grid, args...)
    w3 = getvalue(ψ, i, j, k + stencil_offset(bias, 0, 2), grid, args...)

    β1 = smoothness_indicator((w1, w2, w3), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3))

    # Slide for stencil 1
    w3 = w2
    w2 = w1
    w1 = getvalue(ψ, i, j, k + stencil_offset(bias, 1, 0), grid, args...)

    β2 = smoothness_indicator((w1, w2, w3), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3))

    # Slide for stencil 2
    w3 = w2
    w2 = w1
    w1 = getvalue(ψ, i, j, k + stencil_offset(bias, 2, 0), grid, args...)

    β3 = smoothness_indicator((w1, w2, w3), scheme, Val(2))
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3))

    τ = global_smoothness_indicator(Val(3), (β1, β2, β3))
    α = zweno_alpha_loop(scheme, (β1, β2, β3), τ)
    Σα = α[1] + α[2] + α[3]

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3) / Σα
end

#####
##### WENO{4} (7th order) - x direction
#####

@inline function fused_biased_interpolate_xᶠᵃᵃ(i, j, k, grid,
                                                 scheme::WENO{4, FT}, bias,
                                                 ψ, args...) where FT
    # Stencil 0: load initial window
    w1 = getvalue(ψ, i + stencil_offset(bias, 0, 0), j, k, grid, args...)
    w2 = getvalue(ψ, i + stencil_offset(bias, 0, 1), j, k, grid, args...)
    w3 = getvalue(ψ, i + stencil_offset(bias, 0, 2), j, k, grid, args...)
    w4 = getvalue(ψ, i + stencil_offset(bias, 0, 3), j, k, grid, args...)

    β1 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4))

    # Slide for stencil 1
    w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 1, 0), j, k, grid, args...)

    β2 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4))

    # Slide for stencil 2
    w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 2, 0), j, k, grid, args...)

    β3 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(2))
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4))

    # Slide for stencil 3
    w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 3, 0), j, k, grid, args...)

    β4 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(3))
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4))

    # Compute weights
    τ = global_smoothness_indicator(Val(4), (β1, β2, β3, β4))
    α = zweno_alpha_loop(scheme, (β1, β2, β3, β4), τ)
    Σα = α[1] + α[2] + α[3] + α[4]

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3 + α[4] * p4) / Σα
end

#####
##### WENO{4} (7th order) - y direction
#####

@inline function fused_biased_interpolate_yᵃᶠᵃ(i, j, k, grid,
                                                 scheme::WENO{4, FT}, bias,
                                                 ψ, args...) where FT
    # Stencil 0: load initial window
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 0, 0), k, grid, args...)
    w2 = getvalue(ψ, i, j + stencil_offset(bias, 0, 1), k, grid, args...)
    w3 = getvalue(ψ, i, j + stencil_offset(bias, 0, 2), k, grid, args...)
    w4 = getvalue(ψ, i, j + stencil_offset(bias, 0, 3), k, grid, args...)

    β1 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4))

    # Slide for stencil 1
    w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 1, 0), k, grid, args...)

    β2 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4))

    # Slide for stencil 2
    w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 2, 0), k, grid, args...)

    β3 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(2))
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4))

    # Slide for stencil 3
    w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 3, 0), k, grid, args...)

    β4 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(3))
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4))

    τ = global_smoothness_indicator(Val(4), (β1, β2, β3, β4))
    α = zweno_alpha_loop(scheme, (β1, β2, β3, β4), τ)
    Σα = α[1] + α[2] + α[3] + α[4]

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3 + α[4] * p4) / Σα
end

#####
##### WENO{4} (7th order) - z direction
#####

@inline function fused_biased_interpolate_zᵃᵃᶠ(i, j, k, grid,
                                                 scheme::WENO{4, FT}, bias,
                                                 ψ, args...) where FT
    # Stencil 0: load initial window
    w1 = getvalue(ψ, i, j, k + stencil_offset(bias, 0, 0), grid, args...)
    w2 = getvalue(ψ, i, j, k + stencil_offset(bias, 0, 1), grid, args...)
    w3 = getvalue(ψ, i, j, k + stencil_offset(bias, 0, 2), grid, args...)
    w4 = getvalue(ψ, i, j, k + stencil_offset(bias, 0, 3), grid, args...)

    β1 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4))

    # Slide for stencil 1
    w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j, k + stencil_offset(bias, 1, 0), grid, args...)

    β2 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4))

    # Slide for stencil 2
    w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j, k + stencil_offset(bias, 2, 0), grid, args...)

    β3 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(2))
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4))

    # Slide for stencil 3
    w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j, k + stencil_offset(bias, 3, 0), grid, args...)

    β4 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(3))
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4))

    τ = global_smoothness_indicator(Val(4), (β1, β2, β3, β4))
    α = zweno_alpha_loop(scheme, (β1, β2, β3, β4), τ)
    Σα = α[1] + α[2] + α[3] + α[4]

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3 + α[4] * p4) / Σα
end

#####
##### WENO{5} (9th order) - x direction
#####

@inline function fused_biased_interpolate_xᶠᵃᵃ(i, j, k, grid,
                                                 scheme::WENO{5, FT}, bias,
                                                 ψ, args...) where FT
    # Stencil 0: load initial window
    w1 = getvalue(ψ, i + stencil_offset(bias, 0, 0), j, k, grid, args...)
    w2 = getvalue(ψ, i + stencil_offset(bias, 0, 1), j, k, grid, args...)
    w3 = getvalue(ψ, i + stencil_offset(bias, 0, 2), j, k, grid, args...)
    w4 = getvalue(ψ, i + stencil_offset(bias, 0, 3), j, k, grid, args...)
    w5 = getvalue(ψ, i + stencil_offset(bias, 0, 4), j, k, grid, args...)

    β1 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4, w5))

    # Slide for stencil 1
    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 1, 0), j, k, grid, args...)

    β2 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4, w5))

    # Slide for stencil 2
    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 2, 0), j, k, grid, args...)

    β3 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(2))
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4, w5))

    # Slide for stencil 3
    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 3, 0), j, k, grid, args...)

    β4 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(3))
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4, w5))

    # Slide for stencil 4
    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 4, 0), j, k, grid, args...)

    β5 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(4))
    p5 = biased_p(scheme, bias, Val(4), (w1, w2, w3, w4, w5))

    τ = global_smoothness_indicator(Val(5), (β1, β2, β3, β4, β5))
    α = zweno_alpha_loop(scheme, (β1, β2, β3, β4, β5), τ)
    Σα = α[1] + α[2] + α[3] + α[4] + α[5]

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3 + α[4] * p4 + α[5] * p5) / Σα
end

#####
##### WENO{5} (9th order) - y direction
#####

@inline function fused_biased_interpolate_yᵃᶠᵃ(i, j, k, grid,
                                                 scheme::WENO{5, FT}, bias,
                                                 ψ, args...) where FT
    # Stencil 0: load initial window
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 0, 0), k, grid, args...)
    w2 = getvalue(ψ, i, j + stencil_offset(bias, 0, 1), k, grid, args...)
    w3 = getvalue(ψ, i, j + stencil_offset(bias, 0, 2), k, grid, args...)
    w4 = getvalue(ψ, i, j + stencil_offset(bias, 0, 3), k, grid, args...)
    w5 = getvalue(ψ, i, j + stencil_offset(bias, 0, 4), k, grid, args...)

    β1 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4, w5))

    # Slide for stencil 1
    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 1, 0), k, grid, args...)

    β2 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4, w5))

    # Slide for stencil 2
    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 2, 0), k, grid, args...)

    β3 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(2))
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4, w5))

    # Slide for stencil 3
    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 3, 0), k, grid, args...)

    β4 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(3))
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4, w5))

    # Slide for stencil 4
    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 4, 0), k, grid, args...)

    β5 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(4))
    p5 = biased_p(scheme, bias, Val(4), (w1, w2, w3, w4, w5))

    τ = global_smoothness_indicator(Val(5), (β1, β2, β3, β4, β5))
    α = zweno_alpha_loop(scheme, (β1, β2, β3, β4, β5), τ)
    Σα = α[1] + α[2] + α[3] + α[4] + α[5]

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3 + α[4] * p4 + α[5] * p5) / Σα
end

#####
##### WENO{5} (9th order) - z direction
#####

@inline function fused_biased_interpolate_zᵃᵃᶠ(i, j, k, grid,
                                                 scheme::WENO{5, FT}, bias,
                                                 ψ, args...) where FT
    # Stencil 0: load initial window
    w1 = getvalue(ψ, i, j, k + stencil_offset(bias, 0, 0), grid, args...)
    w2 = getvalue(ψ, i, j, k + stencil_offset(bias, 0, 1), grid, args...)
    w3 = getvalue(ψ, i, j, k + stencil_offset(bias, 0, 2), grid, args...)
    w4 = getvalue(ψ, i, j, k + stencil_offset(bias, 0, 3), grid, args...)
    w5 = getvalue(ψ, i, j, k + stencil_offset(bias, 0, 4), grid, args...)

    β1 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4, w5))

    # Slide for stencil 1
    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j, k + stencil_offset(bias, 1, 0), grid, args...)

    β2 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4, w5))

    # Slide for stencil 2
    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j, k + stencil_offset(bias, 2, 0), grid, args...)

    β3 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(2))
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4, w5))

    # Slide for stencil 3
    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j, k + stencil_offset(bias, 3, 0), grid, args...)

    β4 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(3))
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4, w5))

    # Slide for stencil 4
    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j, k + stencil_offset(bias, 4, 0), grid, args...)

    β5 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(4))
    p5 = biased_p(scheme, bias, Val(4), (w1, w2, w3, w4, w5))

    τ = global_smoothness_indicator(Val(5), (β1, β2, β3, β4, β5))
    α = zweno_alpha_loop(scheme, (β1, β2, β3, β4, β5), τ)
    Σα = α[1] + α[2] + α[3] + α[4] + α[5]

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3 + α[4] * p4 + α[5] * p5) / Σα
end

#####
##### WENO{6} (11th order) - x direction
#####

@inline function fused_biased_interpolate_xᶠᵃᵃ(i, j, k, grid,
                                                 scheme::WENO{6, FT}, bias,
                                                 ψ, args...) where FT
    # Stencil 0: load initial window
    w1 = getvalue(ψ, i + stencil_offset(bias, 0, 0), j, k, grid, args...)
    w2 = getvalue(ψ, i + stencil_offset(bias, 0, 1), j, k, grid, args...)
    w3 = getvalue(ψ, i + stencil_offset(bias, 0, 2), j, k, grid, args...)
    w4 = getvalue(ψ, i + stencil_offset(bias, 0, 3), j, k, grid, args...)
    w5 = getvalue(ψ, i + stencil_offset(bias, 0, 4), j, k, grid, args...)
    w6 = getvalue(ψ, i + stencil_offset(bias, 0, 5), j, k, grid, args...)

    β1 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4, w5, w6))

    # Slide for stencil 1
    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 1, 0), j, k, grid, args...)

    β2 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4, w5, w6))

    # Slide for stencil 2
    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 2, 0), j, k, grid, args...)

    β3 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(2))
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4, w5, w6))

    # Slide for stencil 3
    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 3, 0), j, k, grid, args...)

    β4 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(3))
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4, w5, w6))

    # Slide for stencil 4
    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 4, 0), j, k, grid, args...)

    β5 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(4))
    p5 = biased_p(scheme, bias, Val(4), (w1, w2, w3, w4, w5, w6))

    # Slide for stencil 5
    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i + stencil_offset(bias, 5, 0), j, k, grid, args...)

    β6 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(5))
    p6 = biased_p(scheme, bias, Val(5), (w1, w2, w3, w4, w5, w6))

    τ = global_smoothness_indicator(Val(6), (β1, β2, β3, β4, β5, β6))
    α = zweno_alpha_loop(scheme, (β1, β2, β3, β4, β5, β6), τ)
    Σα = α[1] + α[2] + α[3] + α[4] + α[5] + α[6]

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3 + α[4] * p4 + α[5] * p5 + α[6] * p6) / Σα
end

#####
##### WENO{6} (11th order) - y direction
#####

@inline function fused_biased_interpolate_yᵃᶠᵃ(i, j, k, grid,
                                                 scheme::WENO{6, FT}, bias,
                                                 ψ, args...) where FT
    # Stencil 0: load initial window
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 0, 0), k, grid, args...)
    w2 = getvalue(ψ, i, j + stencil_offset(bias, 0, 1), k, grid, args...)
    w3 = getvalue(ψ, i, j + stencil_offset(bias, 0, 2), k, grid, args...)
    w4 = getvalue(ψ, i, j + stencil_offset(bias, 0, 3), k, grid, args...)
    w5 = getvalue(ψ, i, j + stencil_offset(bias, 0, 4), k, grid, args...)
    w6 = getvalue(ψ, i, j + stencil_offset(bias, 0, 5), k, grid, args...)

    β1 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4, w5, w6))

    # Slide for stencil 1
    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 1, 0), k, grid, args...)

    β2 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4, w5, w6))

    # Slide for stencil 2
    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 2, 0), k, grid, args...)

    β3 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(2))
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4, w5, w6))

    # Slide for stencil 3
    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 3, 0), k, grid, args...)

    β4 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(3))
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4, w5, w6))

    # Slide for stencil 4
    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 4, 0), k, grid, args...)

    β5 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(4))
    p5 = biased_p(scheme, bias, Val(4), (w1, w2, w3, w4, w5, w6))

    # Slide for stencil 5
    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j + stencil_offset(bias, 5, 0), k, grid, args...)

    β6 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(5))
    p6 = biased_p(scheme, bias, Val(5), (w1, w2, w3, w4, w5, w6))

    τ = global_smoothness_indicator(Val(6), (β1, β2, β3, β4, β5, β6))
    α = zweno_alpha_loop(scheme, (β1, β2, β3, β4, β5, β6), τ)
    Σα = α[1] + α[2] + α[3] + α[4] + α[5] + α[6]

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3 + α[4] * p4 + α[5] * p5 + α[6] * p6) / Σα
end

#####
##### WENO{6} (11th order) - z direction
#####

@inline function fused_biased_interpolate_zᵃᵃᶠ(i, j, k, grid,
                                                 scheme::WENO{6, FT}, bias,
                                                 ψ, args...) where FT
    # Stencil 0: load initial window
    w1 = getvalue(ψ, i, j, k + stencil_offset(bias, 0, 0), grid, args...)
    w2 = getvalue(ψ, i, j, k + stencil_offset(bias, 0, 1), grid, args...)
    w3 = getvalue(ψ, i, j, k + stencil_offset(bias, 0, 2), grid, args...)
    w4 = getvalue(ψ, i, j, k + stencil_offset(bias, 0, 3), grid, args...)
    w5 = getvalue(ψ, i, j, k + stencil_offset(bias, 0, 4), grid, args...)
    w6 = getvalue(ψ, i, j, k + stencil_offset(bias, 0, 5), grid, args...)

    β1 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4, w5, w6))

    # Slide for stencil 1
    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j, k + stencil_offset(bias, 1, 0), grid, args...)

    β2 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4, w5, w6))

    # Slide for stencil 2
    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j, k + stencil_offset(bias, 2, 0), grid, args...)

    β3 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(2))
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4, w5, w6))

    # Slide for stencil 3
    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j, k + stencil_offset(bias, 3, 0), grid, args...)

    β4 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(3))
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4, w5, w6))

    # Slide for stencil 4
    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j, k + stencil_offset(bias, 4, 0), grid, args...)

    β5 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(4))
    p5 = biased_p(scheme, bias, Val(4), (w1, w2, w3, w4, w5, w6))

    # Slide for stencil 5
    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = getvalue(ψ, i, j, k + stencil_offset(bias, 5, 0), grid, args...)

    β6 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(5))
    p6 = biased_p(scheme, bias, Val(5), (w1, w2, w3, w4, w5, w6))

    τ = global_smoothness_indicator(Val(6), (β1, β2, β3, β4, β5, β6))
    α = zweno_alpha_loop(scheme, (β1, β2, β3, β4, β5, β6), τ)
    Σα = α[1] + α[2] + α[3] + α[4] + α[5] + α[6]

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3 + α[4] * p4 + α[5] * p5 + α[6] * p6) / Σα
end
