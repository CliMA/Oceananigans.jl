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
##### WENO{2} (3rd order) - x direction
#####

@inline function fused_biased_interpolate_xᶠᵃᵃ(i, j, k, grid,
                                                 scheme::WENO{2, FT}, bias::LeftBias,
                                                 ψ, args...) where FT
    # Stencil 0: w = (ψ[i-1], ψ[i])
    w1 = @inbounds ψ[i - 1, j, k]
    w2 = @inbounds ψ[i,     j, k]

    β1 = smoothness_indicator((w1, w2), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2))

    # Slide for stencil 1: w = (ψ[i-2], ψ[i-1])
    w2 = w1
    w1 = @inbounds ψ[i - 2, j, k]

    β2 = smoothness_indicator((w1, w2), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2))

    # Compute weights
    τ = global_smoothness_indicator(Val(2), (β1, β2))
    α = zweno_alpha_loop(scheme, (β1, β2), τ)
    Σα = α[1] + α[2]

    return @muladd (α[1] * p1 + α[2] * p2) / Σα
end

@inline function fused_biased_interpolate_xᶠᵃᵃ(i, j, k, grid,
                                                 scheme::WENO{2, FT}, bias::RightBias,
                                                 ψ, args...) where FT
    # Stencil 0: w = (ψ[i], ψ[i-1])
    w1 = @inbounds ψ[i,     j, k]
    w2 = @inbounds ψ[i - 1, j, k]

    β1 = smoothness_indicator((w1, w2), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2))

    # Slide for stencil 1: w = (ψ[i+1], ψ[i])
    w2 = w1
    w1 = @inbounds ψ[i + 1, j, k]

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
                                                 scheme::WENO{2, FT}, bias::LeftBias,
                                                 ψ, args...) where FT
    w1 = @inbounds ψ[i, j - 1, k]
    w2 = @inbounds ψ[i, j,     k]

    β1 = smoothness_indicator((w1, w2), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2))

    w2 = w1
    w1 = @inbounds ψ[i, j - 2, k]

    β2 = smoothness_indicator((w1, w2), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2))

    τ = global_smoothness_indicator(Val(2), (β1, β2))
    α = zweno_alpha_loop(scheme, (β1, β2), τ)
    Σα = α[1] + α[2]

    return @muladd (α[1] * p1 + α[2] * p2) / Σα
end

@inline function fused_biased_interpolate_yᵃᶠᵃ(i, j, k, grid,
                                                 scheme::WENO{2, FT}, bias::RightBias,
                                                 ψ, args...) where FT
    w1 = @inbounds ψ[i, j,     k]
    w2 = @inbounds ψ[i, j - 1, k]

    β1 = smoothness_indicator((w1, w2), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2))

    w2 = w1
    w1 = @inbounds ψ[i, j + 1, k]

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
                                                 scheme::WENO{2, FT}, bias::LeftBias,
                                                 ψ, args...) where FT
    w1 = @inbounds ψ[i, j, k - 1]
    w2 = @inbounds ψ[i, j, k    ]

    β1 = smoothness_indicator((w1, w2), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2))

    w2 = w1
    w1 = @inbounds ψ[i, j, k - 2]

    β2 = smoothness_indicator((w1, w2), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2))

    τ = global_smoothness_indicator(Val(2), (β1, β2))
    α = zweno_alpha_loop(scheme, (β1, β2), τ)
    Σα = α[1] + α[2]

    return @muladd (α[1] * p1 + α[2] * p2) / Σα
end

@inline function fused_biased_interpolate_zᵃᵃᶠ(i, j, k, grid,
                                                 scheme::WENO{2, FT}, bias::RightBias,
                                                 ψ, args...) where FT
    w1 = @inbounds ψ[i, j, k    ]
    w2 = @inbounds ψ[i, j, k - 1]

    β1 = smoothness_indicator((w1, w2), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2))

    w2 = w1
    w1 = @inbounds ψ[i, j, k + 1]

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
                                                 scheme::WENO{3, FT}, bias::LeftBias,
                                                 ψ, args...) where FT
    # Stencil 0: w = (ψ[i-1], ψ[i], ψ[i+1])
    w1 = @inbounds ψ[i - 1, j, k]
    w2 = @inbounds ψ[i,     j, k]
    w3 = @inbounds ψ[i + 1, j, k]

    β1 = smoothness_indicator((w1, w2, w3), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3))

    # Slide for stencil 1: w = (ψ[i-2], ψ[i-1], ψ[i])
    w3 = w2
    w2 = w1
    w1 = @inbounds ψ[i - 2, j, k]

    β2 = smoothness_indicator((w1, w2, w3), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3))

    # Slide for stencil 2: w = (ψ[i-3], ψ[i-2], ψ[i-1])
    w3 = w2
    w2 = w1
    w1 = @inbounds ψ[i - 3, j, k]

    β3 = smoothness_indicator((w1, w2, w3), scheme, Val(2))
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3))

    # Compute weights
    τ = global_smoothness_indicator(Val(3), (β1, β2, β3))
    α = zweno_alpha_loop(scheme, (β1, β2, β3), τ)
    Σα = α[1] + α[2] + α[3]

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3) / Σα
end

@inline function fused_biased_interpolate_xᶠᵃᵃ(i, j, k, grid,
                                                 scheme::WENO{3, FT}, bias::RightBias,
                                                 ψ, args...) where FT
    # Stencil 0: w = (ψ[i], ψ[i-1], ψ[i-2])
    w1 = @inbounds ψ[i,     j, k]
    w2 = @inbounds ψ[i - 1, j, k]
    w3 = @inbounds ψ[i - 2, j, k]

    β1 = smoothness_indicator((w1, w2, w3), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3))

    # Slide for stencil 1: w = (ψ[i+1], ψ[i], ψ[i-1])
    w3 = w2
    w2 = w1
    w1 = @inbounds ψ[i + 1, j, k]

    β2 = smoothness_indicator((w1, w2, w3), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3))

    # Slide for stencil 2: w = (ψ[i+2], ψ[i+1], ψ[i])
    w3 = w2
    w2 = w1
    w1 = @inbounds ψ[i + 2, j, k]

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
                                                 scheme::WENO{3, FT}, bias::LeftBias,
                                                 ψ, args...) where FT
    w1 = @inbounds ψ[i, j - 1, k]
    w2 = @inbounds ψ[i, j,     k]
    w3 = @inbounds ψ[i, j + 1, k]

    β1 = smoothness_indicator((w1, w2, w3), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3))

    w3 = w2
    w2 = w1
    w1 = @inbounds ψ[i, j - 2, k]

    β2 = smoothness_indicator((w1, w2, w3), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3))

    w3 = w2
    w2 = w1
    w1 = @inbounds ψ[i, j - 3, k]

    β3 = smoothness_indicator((w1, w2, w3), scheme, Val(2))
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3))

    τ = global_smoothness_indicator(Val(3), (β1, β2, β3))
    α = zweno_alpha_loop(scheme, (β1, β2, β3), τ)
    Σα = α[1] + α[2] + α[3]

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3) / Σα
end

@inline function fused_biased_interpolate_yᵃᶠᵃ(i, j, k, grid,
                                                 scheme::WENO{3, FT}, bias::RightBias,
                                                 ψ, args...) where FT
    w1 = @inbounds ψ[i, j,     k]
    w2 = @inbounds ψ[i, j - 1, k]
    w3 = @inbounds ψ[i, j - 2, k]

    β1 = smoothness_indicator((w1, w2, w3), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3))

    w3 = w2
    w2 = w1
    w1 = @inbounds ψ[i, j + 1, k]

    β2 = smoothness_indicator((w1, w2, w3), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3))

    w3 = w2
    w2 = w1
    w1 = @inbounds ψ[i, j + 2, k]

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
                                                 scheme::WENO{3, FT}, bias::LeftBias,
                                                 ψ, args...) where FT
    w1 = @inbounds ψ[i, j, k - 1]
    w2 = @inbounds ψ[i, j, k    ]
    w3 = @inbounds ψ[i, j, k + 1]

    β1 = smoothness_indicator((w1, w2, w3), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3))

    w3 = w2
    w2 = w1
    w1 = @inbounds ψ[i, j, k - 2]

    β2 = smoothness_indicator((w1, w2, w3), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3))

    w3 = w2
    w2 = w1
    w1 = @inbounds ψ[i, j, k - 3]

    β3 = smoothness_indicator((w1, w2, w3), scheme, Val(2))
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3))

    τ = global_smoothness_indicator(Val(3), (β1, β2, β3))
    α = zweno_alpha_loop(scheme, (β1, β2, β3), τ)
    Σα = α[1] + α[2] + α[3]

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3) / Σα
end

@inline function fused_biased_interpolate_zᵃᵃᶠ(i, j, k, grid,
                                                 scheme::WENO{3, FT}, bias::RightBias,
                                                 ψ, args...) where FT
    w1 = @inbounds ψ[i, j, k    ]
    w2 = @inbounds ψ[i, j, k - 1]
    w3 = @inbounds ψ[i, j, k - 2]

    β1 = smoothness_indicator((w1, w2, w3), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3))

    w3 = w2
    w2 = w1
    w1 = @inbounds ψ[i, j, k + 1]

    β2 = smoothness_indicator((w1, w2, w3), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3))

    w3 = w2
    w2 = w1
    w1 = @inbounds ψ[i, j, k + 2]

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
                                                 scheme::WENO{4, FT}, bias::LeftBias,
                                                 ψ, args...) where FT
    # Stencil 0: w = (ψ[i-1], ψ[i], ψ[i+1], ψ[i+2])
    w1 = @inbounds ψ[i - 1, j, k]
    w2 = @inbounds ψ[i,     j, k]
    w3 = @inbounds ψ[i + 1, j, k]
    w4 = @inbounds ψ[i + 2, j, k]

    β1 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4))

    # Slide for stencil 1
    w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i - 2, j, k]

    β2 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4))

    # Slide for stencil 2
    w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i - 3, j, k]

    β3 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(2))
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4))

    # Slide for stencil 3
    w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i - 4, j, k]

    β4 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(3))
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4))

    # Compute weights
    τ = global_smoothness_indicator(Val(4), (β1, β2, β3, β4))
    α = zweno_alpha_loop(scheme, (β1, β2, β3, β4), τ)
    Σα = α[1] + α[2] + α[3] + α[4]

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3 + α[4] * p4) / Σα
end

@inline function fused_biased_interpolate_xᶠᵃᵃ(i, j, k, grid,
                                                 scheme::WENO{4, FT}, bias::RightBias,
                                                 ψ, args...) where FT
    # Stencil 0: w = (ψ[i], ψ[i-1], ψ[i-2], ψ[i-3])
    w1 = @inbounds ψ[i,     j, k]
    w2 = @inbounds ψ[i - 1, j, k]
    w3 = @inbounds ψ[i - 2, j, k]
    w4 = @inbounds ψ[i - 3, j, k]

    β1 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4))

    # Slide for stencil 1
    w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i + 1, j, k]

    β2 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4))

    # Slide for stencil 2
    w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i + 2, j, k]

    β3 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(2))
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4))

    # Slide for stencil 3
    w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i + 3, j, k]

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
                                                 scheme::WENO{4, FT}, bias::LeftBias,
                                                 ψ, args...) where FT
    w1 = @inbounds ψ[i, j - 1, k]
    w2 = @inbounds ψ[i, j,     k]
    w3 = @inbounds ψ[i, j + 1, k]
    w4 = @inbounds ψ[i, j + 2, k]

    β1 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4))

    w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j - 2, k]

    β2 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4))

    w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j - 3, k]

    β3 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(2))
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4))

    w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j - 4, k]

    β4 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(3))
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4))

    τ = global_smoothness_indicator(Val(4), (β1, β2, β3, β4))
    α = zweno_alpha_loop(scheme, (β1, β2, β3, β4), τ)
    Σα = α[1] + α[2] + α[3] + α[4]

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3 + α[4] * p4) / Σα
end

@inline function fused_biased_interpolate_yᵃᶠᵃ(i, j, k, grid,
                                                 scheme::WENO{4, FT}, bias::RightBias,
                                                 ψ, args...) where FT
    w1 = @inbounds ψ[i, j,     k]
    w2 = @inbounds ψ[i, j - 1, k]
    w3 = @inbounds ψ[i, j - 2, k]
    w4 = @inbounds ψ[i, j - 3, k]

    β1 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4))

    w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j + 1, k]

    β2 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4))

    w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j + 2, k]

    β3 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(2))
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4))

    w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j + 3, k]

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
                                                 scheme::WENO{4, FT}, bias::LeftBias,
                                                 ψ, args...) where FT
    w1 = @inbounds ψ[i, j, k - 1]
    w2 = @inbounds ψ[i, j, k    ]
    w3 = @inbounds ψ[i, j, k + 1]
    w4 = @inbounds ψ[i, j, k + 2]

    β1 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4))

    w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j, k - 2]

    β2 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4))

    w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j, k - 3]

    β3 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(2))
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4))

    w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j, k - 4]

    β4 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(3))
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4))

    τ = global_smoothness_indicator(Val(4), (β1, β2, β3, β4))
    α = zweno_alpha_loop(scheme, (β1, β2, β3, β4), τ)
    Σα = α[1] + α[2] + α[3] + α[4]

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3 + α[4] * p4) / Σα
end

@inline function fused_biased_interpolate_zᵃᵃᶠ(i, j, k, grid,
                                                 scheme::WENO{4, FT}, bias::RightBias,
                                                 ψ, args...) where FT
    w1 = @inbounds ψ[i, j, k    ]
    w2 = @inbounds ψ[i, j, k - 1]
    w3 = @inbounds ψ[i, j, k - 2]
    w4 = @inbounds ψ[i, j, k - 3]

    β1 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4))

    w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j, k + 1]

    β2 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4))

    w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j, k + 2]

    β3 = smoothness_indicator((w1, w2, w3, w4), scheme, Val(2))
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4))

    w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j, k + 3]

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
                                                 scheme::WENO{5, FT}, bias::LeftBias,
                                                 ψ, args...) where FT
    w1 = @inbounds ψ[i - 1, j, k]
    w2 = @inbounds ψ[i,     j, k]
    w3 = @inbounds ψ[i + 1, j, k]
    w4 = @inbounds ψ[i + 2, j, k]
    w5 = @inbounds ψ[i + 3, j, k]

    β1 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i - 2, j, k]

    β2 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i - 3, j, k]

    β3 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(2))
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i - 4, j, k]

    β4 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(3))
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i - 5, j, k]

    β5 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(4))
    p5 = biased_p(scheme, bias, Val(4), (w1, w2, w3, w4, w5))

    τ = global_smoothness_indicator(Val(5), (β1, β2, β3, β4, β5))
    α = zweno_alpha_loop(scheme, (β1, β2, β3, β4, β5), τ)
    Σα = α[1] + α[2] + α[3] + α[4] + α[5]

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3 + α[4] * p4 + α[5] * p5) / Σα
end

@inline function fused_biased_interpolate_xᶠᵃᵃ(i, j, k, grid,
                                                 scheme::WENO{5, FT}, bias::RightBias,
                                                 ψ, args...) where FT
    w1 = @inbounds ψ[i,     j, k]
    w2 = @inbounds ψ[i - 1, j, k]
    w3 = @inbounds ψ[i - 2, j, k]
    w4 = @inbounds ψ[i - 3, j, k]
    w5 = @inbounds ψ[i - 4, j, k]

    β1 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i + 1, j, k]

    β2 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i + 2, j, k]

    β3 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(2))
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i + 3, j, k]

    β4 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(3))
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i + 4, j, k]

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
                                                 scheme::WENO{5, FT}, bias::LeftBias,
                                                 ψ, args...) where FT
    w1 = @inbounds ψ[i, j - 1, k]
    w2 = @inbounds ψ[i, j,     k]
    w3 = @inbounds ψ[i, j + 1, k]
    w4 = @inbounds ψ[i, j + 2, k]
    w5 = @inbounds ψ[i, j + 3, k]

    β1 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j - 2, k]

    β2 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j - 3, k]

    β3 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(2))
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j - 4, k]

    β4 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(3))
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j - 5, k]

    β5 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(4))
    p5 = biased_p(scheme, bias, Val(4), (w1, w2, w3, w4, w5))

    τ = global_smoothness_indicator(Val(5), (β1, β2, β3, β4, β5))
    α = zweno_alpha_loop(scheme, (β1, β2, β3, β4, β5), τ)
    Σα = α[1] + α[2] + α[3] + α[4] + α[5]

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3 + α[4] * p4 + α[5] * p5) / Σα
end

@inline function fused_biased_interpolate_yᵃᶠᵃ(i, j, k, grid,
                                                 scheme::WENO{5, FT}, bias::RightBias,
                                                 ψ, args...) where FT
    w1 = @inbounds ψ[i, j,     k]
    w2 = @inbounds ψ[i, j - 1, k]
    w3 = @inbounds ψ[i, j - 2, k]
    w4 = @inbounds ψ[i, j - 3, k]
    w5 = @inbounds ψ[i, j - 4, k]

    β1 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j + 1, k]

    β2 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j + 2, k]

    β3 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(2))
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j + 3, k]

    β4 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(3))
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j + 4, k]

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
                                                 scheme::WENO{5, FT}, bias::LeftBias,
                                                 ψ, args...) where FT
    w1 = @inbounds ψ[i, j, k - 1]
    w2 = @inbounds ψ[i, j, k    ]
    w3 = @inbounds ψ[i, j, k + 1]
    w4 = @inbounds ψ[i, j, k + 2]
    w5 = @inbounds ψ[i, j, k + 3]

    β1 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j, k - 2]

    β2 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j, k - 3]

    β3 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(2))
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j, k - 4]

    β4 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(3))
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j, k - 5]

    β5 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(4))
    p5 = biased_p(scheme, bias, Val(4), (w1, w2, w3, w4, w5))

    τ = global_smoothness_indicator(Val(5), (β1, β2, β3, β4, β5))
    α = zweno_alpha_loop(scheme, (β1, β2, β3, β4, β5), τ)
    Σα = α[1] + α[2] + α[3] + α[4] + α[5]

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3 + α[4] * p4 + α[5] * p5) / Σα
end

@inline function fused_biased_interpolate_zᵃᵃᶠ(i, j, k, grid,
                                               scheme::WENO{5, FT}, bias::RightBias,
                                               ψ, args...) where FT
    w1 = @inbounds ψ[i, j, k    ]
    w2 = @inbounds ψ[i, j, k - 1]
    w3 = @inbounds ψ[i, j, k - 2]
    w4 = @inbounds ψ[i, j, k - 3]
    w5 = @inbounds ψ[i, j, k - 4]

    β1 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j, k + 1]

    β2 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j, k + 2]

    β3 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(2))
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j, k + 3]

    β4 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(3))
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j, k + 4]

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
                                                 scheme::WENO{6, FT}, bias::LeftBias,
                                                 ψ, args...) where FT
    w1 = @inbounds ψ[i - 1, j, k]
    w2 = @inbounds ψ[i,     j, k]
    w3 = @inbounds ψ[i + 1, j, k]
    w4 = @inbounds ψ[i + 2, j, k]
    w5 = @inbounds ψ[i + 3, j, k]
    w6 = @inbounds ψ[i + 4, j, k]

    β1 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i - 2, j, k]

    β2 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i - 3, j, k]

    β3 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(2))
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i - 4, j, k]

    β4 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(3))
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i - 5, j, k]

    β5 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(4))
    p5 = biased_p(scheme, bias, Val(4), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i - 6, j, k]

    β6 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(5))
    p6 = biased_p(scheme, bias, Val(5), (w1, w2, w3, w4, w5, w6))

    τ = global_smoothness_indicator(Val(6), (β1, β2, β3, β4, β5, β6))
    α = zweno_alpha_loop(scheme, (β1, β2, β3, β4, β5, β6), τ)
    Σα = α[1] + α[2] + α[3] + α[4] + α[5] + α[6]

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3 + α[4] * p4 + α[5] * p5 + α[6] * p6) / Σα
end

@inline function fused_biased_interpolate_xᶠᵃᵃ(i, j, k, grid,
                                                 scheme::WENO{6, FT}, bias::RightBias,
                                                 ψ, args...) where FT
    w1 = @inbounds ψ[i,     j, k]
    w2 = @inbounds ψ[i - 1, j, k]
    w3 = @inbounds ψ[i - 2, j, k]
    w4 = @inbounds ψ[i - 3, j, k]
    w5 = @inbounds ψ[i - 4, j, k]
    w6 = @inbounds ψ[i - 5, j, k]

    β1 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i + 1, j, k]

    β2 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i + 2, j, k]

    β3 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(2))
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i + 3, j, k]

    β4 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(3))
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i + 4, j, k]

    β5 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(4))
    p5 = biased_p(scheme, bias, Val(4), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i + 5, j, k]

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
                                                 scheme::WENO{6, FT}, bias::LeftBias,
                                                 ψ, args...) where FT
    w1 = @inbounds ψ[i, j - 1, k]
    w2 = @inbounds ψ[i, j,     k]
    w3 = @inbounds ψ[i, j + 1, k]
    w4 = @inbounds ψ[i, j + 2, k]
    w5 = @inbounds ψ[i, j + 3, k]
    w6 = @inbounds ψ[i, j + 4, k]

    β1 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j - 2, k]

    β2 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j - 3, k]

    β3 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(2))
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j - 4, k]

    β4 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(3))
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j - 5, k]

    β5 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(4))
    p5 = biased_p(scheme, bias, Val(4), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j - 6, k]

    β6 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(5))
    p6 = biased_p(scheme, bias, Val(5), (w1, w2, w3, w4, w5, w6))

    τ = global_smoothness_indicator(Val(6), (β1, β2, β3, β4, β5, β6))
    α = zweno_alpha_loop(scheme, (β1, β2, β3, β4, β5, β6), τ)
    Σα = α[1] + α[2] + α[3] + α[4] + α[5] + α[6]

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3 + α[4] * p4 + α[5] * p5 + α[6] * p6) / Σα
end

@inline function fused_biased_interpolate_yᵃᶠᵃ(i, j, k, grid,
                                                 scheme::WENO{6, FT}, bias::RightBias,
                                                 ψ, args...) where FT
    w1 = @inbounds ψ[i, j,     k]
    w2 = @inbounds ψ[i, j - 1, k]
    w3 = @inbounds ψ[i, j - 2, k]
    w4 = @inbounds ψ[i, j - 3, k]
    w5 = @inbounds ψ[i, j - 4, k]
    w6 = @inbounds ψ[i, j - 5, k]

    β1 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j + 1, k]

    β2 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j + 2, k]

    β3 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(2))
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j + 3, k]

    β4 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(3))
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j + 4, k]

    β5 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(4))
    p5 = biased_p(scheme, bias, Val(4), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j + 5, k]

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
                                                 scheme::WENO{6, FT}, bias::LeftBias,
                                                 ψ, args...) where FT
    w1 = @inbounds ψ[i, j, k - 1]
    w2 = @inbounds ψ[i, j, k    ]
    w3 = @inbounds ψ[i, j, k + 1]
    w4 = @inbounds ψ[i, j, k + 2]
    w5 = @inbounds ψ[i, j, k + 3]
    w6 = @inbounds ψ[i, j, k + 4]

    β1 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j, k - 2]

    β2 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j, k - 3]

    β3 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(2))
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j, k - 4]

    β4 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(3))
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j, k - 5]

    β5 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(4))
    p5 = biased_p(scheme, bias, Val(4), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j, k - 6]

    β6 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(5))
    p6 = biased_p(scheme, bias, Val(5), (w1, w2, w3, w4, w5, w6))

    τ = global_smoothness_indicator(Val(6), (β1, β2, β3, β4, β5, β6))
    α = zweno_alpha_loop(scheme, (β1, β2, β3, β4, β5, β6), τ)
    Σα = α[1] + α[2] + α[3] + α[4] + α[5] + α[6]

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3 + α[4] * p4 + α[5] * p5 + α[6] * p6) / Σα
end

@inline function fused_biased_interpolate_zᵃᵃᶠ(i, j, k, grid,
                                                 scheme::WENO{6, FT}, bias::RightBias,
                                                 ψ, args...) where FT
    w1 = @inbounds ψ[i, j, k    ]
    w2 = @inbounds ψ[i, j, k - 1]
    w3 = @inbounds ψ[i, j, k - 2]
    w4 = @inbounds ψ[i, j, k - 3]
    w5 = @inbounds ψ[i, j, k - 4]
    w6 = @inbounds ψ[i, j, k - 5]

    β1 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j, k + 1]

    β2 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j, k + 2]

    β3 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(2))
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j, k + 3]

    β4 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(3))
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j, k + 4]

    β5 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(4))
    p5 = biased_p(scheme, bias, Val(4), (w1, w2, w3, w4, w5, w6))

    w6 = w5; w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = @inbounds ψ[i, j, k + 5]

    β6 = smoothness_indicator((w1, w2, w3, w4, w5, w6), scheme, Val(5))
    p6 = biased_p(scheme, bias, Val(5), (w1, w2, w3, w4, w5, w6))

    τ = global_smoothness_indicator(Val(6), (β1, β2, β3, β4, β5, β6))
    α = zweno_alpha_loop(scheme, (β1, β2, β3, β4, β5, β6), τ)
    Σα = α[1] + α[2] + α[3] + α[4] + α[5] + α[6]

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3 + α[4] * p4 + α[5] * p5 + α[6] * p6) / Σα
end

#####
##### Function versions (ψ::Function) for functional reconstruction
##### These are used for vorticity reconstruction in VectorInvariant advection
#####

##### WENO{5} Function - x direction (most common for vorticity)

@inline function fused_biased_interpolate_xᶠᵃᵃ(i, j, k, grid,
                                                 scheme::WENO{5, FT}, bias::LeftBias,
                                                 ψ::Function, args...) where FT
    w1 = ψ(i - 1, j, k, grid, args...)
    w2 = ψ(i,     j, k, grid, args...)
    w3 = ψ(i + 1, j, k, grid, args...)
    w4 = ψ(i + 2, j, k, grid, args...)
    w5 = ψ(i + 3, j, k, grid, args...)

    β1 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ψ(i - 2, j, k, grid, args...)

    β2 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ψ(i - 3, j, k, grid, args...)

    β3 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(2))
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ψ(i - 4, j, k, grid, args...)

    β4 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(3))
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ψ(i - 5, j, k, grid, args...)

    β5 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(4))
    p5 = biased_p(scheme, bias, Val(4), (w1, w2, w3, w4, w5))

    τ = global_smoothness_indicator(Val(5), (β1, β2, β3, β4, β5))
    α = zweno_alpha_loop(scheme, (β1, β2, β3, β4, β5), τ)
    Σα = α[1] + α[2] + α[3] + α[4] + α[5]

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3 + α[4] * p4 + α[5] * p5) / Σα
end

@inline function fused_biased_interpolate_xᶠᵃᵃ(i, j, k, grid,
                                                 scheme::WENO{5, FT}, bias::RightBias,
                                                 ψ::Function, args...) where FT
    w1 = ψ(i,     j, k, grid, args...)
    w2 = ψ(i - 1, j, k, grid, args...)
    w3 = ψ(i - 2, j, k, grid, args...)
    w4 = ψ(i - 3, j, k, grid, args...)
    w5 = ψ(i - 4, j, k, grid, args...)

    β1 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ψ(i + 1, j, k, grid, args...)

    β2 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ψ(i + 2, j, k, grid, args...)

    β3 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(2))
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ψ(i + 3, j, k, grid, args...)

    β4 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(3))
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ψ(i + 4, j, k, grid, args...)

    β5 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(4))
    p5 = biased_p(scheme, bias, Val(4), (w1, w2, w3, w4, w5))

    τ = global_smoothness_indicator(Val(5), (β1, β2, β3, β4, β5))
    α = zweno_alpha_loop(scheme, (β1, β2, β3, β4, β5), τ)
    Σα = α[1] + α[2] + α[3] + α[4] + α[5]

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3 + α[4] * p4 + α[5] * p5) / Σα
end

##### WENO{5} Function - y direction

@inline function fused_biased_interpolate_yᵃᶠᵃ(i, j, k, grid,
                                                 scheme::WENO{5, FT}, bias::LeftBias,
                                                 ψ::Function, args...) where FT
    w1 = ψ(i, j - 1, k, grid, args...)
    w2 = ψ(i, j,     k, grid, args...)
    w3 = ψ(i, j + 1, k, grid, args...)
    w4 = ψ(i, j + 2, k, grid, args...)
    w5 = ψ(i, j + 3, k, grid, args...)

    β1 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ψ(i, j - 2, k, grid, args...)

    β2 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ψ(i, j - 3, k, grid, args...)

    β3 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(2))
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ψ(i, j - 4, k, grid, args...)

    β4 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(3))
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ψ(i, j - 5, k, grid, args...)

    β5 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(4))
    p5 = biased_p(scheme, bias, Val(4), (w1, w2, w3, w4, w5))

    τ = global_smoothness_indicator(Val(5), (β1, β2, β3, β4, β5))
    α = zweno_alpha_loop(scheme, (β1, β2, β3, β4, β5), τ)
    Σα = α[1] + α[2] + α[3] + α[4] + α[5]

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3 + α[4] * p4 + α[5] * p5) / Σα
end

@inline function fused_biased_interpolate_yᵃᶠᵃ(i, j, k, grid,
                                                 scheme::WENO{5, FT}, bias::RightBias,
                                                 ψ::Function, args...) where FT
    w1 = ψ(i, j,     k, grid, args...)
    w2 = ψ(i, j - 1, k, grid, args...)
    w3 = ψ(i, j - 2, k, grid, args...)
    w4 = ψ(i, j - 3, k, grid, args...)
    w5 = ψ(i, j - 4, k, grid, args...)

    β1 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ψ(i, j + 1, k, grid, args...)

    β2 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ψ(i, j + 2, k, grid, args...)

    β3 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(2))
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ψ(i, j + 3, k, grid, args...)

    β4 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(3))
    p4 = biased_p(scheme, bias, Val(3), (w1, w2, w3, w4, w5))

    w5 = w4; w4 = w3; w3 = w2; w2 = w1
    w1 = ψ(i, j + 4, k, grid, args...)

    β5 = smoothness_indicator((w1, w2, w3, w4, w5), scheme, Val(4))
    p5 = biased_p(scheme, bias, Val(4), (w1, w2, w3, w4, w5))

    τ = global_smoothness_indicator(Val(5), (β1, β2, β3, β4, β5))
    α = zweno_alpha_loop(scheme, (β1, β2, β3, β4, β5), τ)
    Σα = α[1] + α[2] + α[3] + α[4] + α[5]

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3 + α[4] * p4 + α[5] * p5) / Σα
end

##### WENO{3} Function - x direction

@inline function fused_biased_interpolate_xᶠᵃᵃ(i, j, k, grid,
                                                 scheme::WENO{3, FT}, bias::LeftBias,
                                                 ψ::Function, args...) where FT
    w1 = ψ(i - 1, j, k, grid, args...)
    w2 = ψ(i,     j, k, grid, args...)
    w3 = ψ(i + 1, j, k, grid, args...)

    β1 = smoothness_indicator((w1, w2, w3), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3))

    w3 = w2; w2 = w1
    w1 = ψ(i - 2, j, k, grid, args...)

    β2 = smoothness_indicator((w1, w2, w3), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3))

    w3 = w2; w2 = w1
    w1 = ψ(i - 3, j, k, grid, args...)

    β3 = smoothness_indicator((w1, w2, w3), scheme, Val(2))
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3))

    τ = global_smoothness_indicator(Val(3), (β1, β2, β3))
    α = zweno_alpha_loop(scheme, (β1, β2, β3), τ)
    Σα = α[1] + α[2] + α[3]

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3) / Σα
end

@inline function fused_biased_interpolate_xᶠᵃᵃ(i, j, k, grid,
                                                 scheme::WENO{3, FT}, bias::RightBias,
                                                 ψ::Function, args...) where FT
    w1 = ψ(i,     j, k, grid, args...)
    w2 = ψ(i - 1, j, k, grid, args...)
    w3 = ψ(i - 2, j, k, grid, args...)

    β1 = smoothness_indicator((w1, w2, w3), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3))

    w3 = w2; w2 = w1
    w1 = ψ(i + 1, j, k, grid, args...)

    β2 = smoothness_indicator((w1, w2, w3), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3))

    w3 = w2; w2 = w1
    w1 = ψ(i + 2, j, k, grid, args...)

    β3 = smoothness_indicator((w1, w2, w3), scheme, Val(2))
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3))

    τ = global_smoothness_indicator(Val(3), (β1, β2, β3))
    α = zweno_alpha_loop(scheme, (β1, β2, β3), τ)
    Σα = α[1] + α[2] + α[3]

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3) / Σα
end

##### WENO{3} Function - y direction

@inline function fused_biased_interpolate_yᵃᶠᵃ(i, j, k, grid,
                                                 scheme::WENO{3, FT}, bias::LeftBias,
                                                 ψ::Function, args...) where FT
    w1 = ψ(i, j - 1, k, grid, args...)
    w2 = ψ(i, j,     k, grid, args...)
    w3 = ψ(i, j + 1, k, grid, args...)

    β1 = smoothness_indicator((w1, w2, w3), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3))

    w3 = w2; w2 = w1
    w1 = ψ(i, j - 2, k, grid, args...)

    β2 = smoothness_indicator((w1, w2, w3), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3))

    w3 = w2; w2 = w1
    w1 = ψ(i, j - 3, k, grid, args...)

    β3 = smoothness_indicator((w1, w2, w3), scheme, Val(2))
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3))

    τ = global_smoothness_indicator(Val(3), (β1, β2, β3))
    α = zweno_alpha_loop(scheme, (β1, β2, β3), τ)
    Σα = α[1] + α[2] + α[3]

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3) / Σα
end

@inline function fused_biased_interpolate_yᵃᶠᵃ(i, j, k, grid,
                                                 scheme::WENO{3, FT}, bias::RightBias,
                                                 ψ::Function, args...) where FT
    w1 = ψ(i, j,     k, grid, args...)
    w2 = ψ(i, j - 1, k, grid, args...)
    w3 = ψ(i, j - 2, k, grid, args...)

    β1 = smoothness_indicator((w1, w2, w3), scheme, Val(0))
    p1 = biased_p(scheme, bias, Val(0), (w1, w2, w3))

    w3 = w2; w2 = w1
    w1 = ψ(i, j + 1, k, grid, args...)

    β2 = smoothness_indicator((w1, w2, w3), scheme, Val(1))
    p2 = biased_p(scheme, bias, Val(1), (w1, w2, w3))

    w3 = w2; w2 = w1
    w1 = ψ(i, j + 2, k, grid, args...)

    β3 = smoothness_indicator((w1, w2, w3), scheme, Val(2))
    p3 = biased_p(scheme, bias, Val(2), (w1, w2, w3))

    τ = global_smoothness_indicator(Val(3), (β1, β2, β3))
    α = zweno_alpha_loop(scheme, (β1, β2, β3), τ)
    Σα = α[1] + α[2] + α[3]

    return @muladd (α[1] * p1 + α[2] * p2 + α[3] * p3) / Σα
end
