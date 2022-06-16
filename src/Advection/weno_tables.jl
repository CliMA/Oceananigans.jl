## Values taken from Balsara & Shu "Monotonicity Preserving Weighted Essentially Non-oscillatory Schemes with Inceasingly High Order of Accuracy"

# Optimal WENO coefficients
Cw(::WENO{2}, ::Val{0}) = 2/3;   Cw(::WENO{2}, ::Val{1}) = 1/3
Cw(::WENO{3}, ::Val{0}) = 3/10;  Cw(::WENO{3}, ::Val{1}) = 3/5;   Cw(::WENO{3}, ::Val{2}) = 1/10
Cw(::WENO{4}, ::Val{0}) = 1/35;  Cw(::WENO{4}, ::Val{1}) = 12/35; Cw(::WENO{4}, ::Val{2}) = 18/35; Cw(::WENO{4}, ::Val{3}) = 4/35
Cw(::WENO{5}, ::Val{0}) = 1/126; Cw(::WENO{5}, ::Val{1}) = 10/63; Cw(::WENO{5}, ::Val{2}) = 10/21; Cw(::WENO{5}, ::Val{3}) = 20/63;   Cw(::WENO{5}, ::Val{4}) = 5/126
Cw(::WENO{6}, ::Val{0}) = 1/462; Cw(::WENO{6}, ::Val{1}) = 5/77;  Cw(::WENO{6}, ::Val{2}) = 25/77; Cw(::WENO{6}, ::Val{3}) = 100/231; Cw(::WENO{6}, ::Val{4}) = 25/154; Cw(::WENO{6}, ::Val{5}) = 1/77

# ENO reconstruction coefficients (uniform grid)
@inline coeff_left_p(scheme::WENO{2, FT}, ::Val{0}, ::Type{Nothing}, args...) where FT = ( FT(1/2), FT(1/2))
@inline coeff_left_p(scheme::WENO{2, FT}, ::Val{1}, ::Type{Nothing}, args...) where FT = (-FT(1/2), FT(3/2))

@inline coeff_left_p(scheme::WENO{3, FT}, ::Val{0}, ::Type{Nothing}, args...) where FT = ( FT(1/3),  FT(5/6), -FT(1/6))
@inline coeff_left_p(scheme::WENO{3, FT}, ::Val{1}, ::Type{Nothing}, args...) where FT = (-FT(1/6),  FT(5/6),  FT(1/3))
@inline coeff_left_p(scheme::WENO{3, FT}, ::Val{2}, ::Type{Nothing}, args...) where FT = ( FT(1/3), -FT(7/6),  FT(11/6))

@inline coeff_left_p(scheme::WENO{4, FT}, ::Val{0}, ::Type{Nothing}, args...) where FT = ( FT(1/4),   FT(13/12), -FT(5/12),  FT(1/2))
@inline coeff_left_p(scheme::WENO{4, FT}, ::Val{1}, ::Type{Nothing}, args...) where FT = (-FT(1/12),  FT(7/12),   FT(7/12),  FT(1/12))
@inline coeff_left_p(scheme::WENO{4, FT}, ::Val{2}, ::Type{Nothing}, args...) where FT = ( FT(1/12), -FT(5/12),   FT(13/12), FT(1/4))
@inline coeff_left_p(scheme::WENO{4, FT}, ::Val{3}, ::Type{Nothing}, args...) where FT = (-FT(1/4),   FT(13/12), -FT(23/12), FT(25/12))

@inline coeff_left_p(scheme::WENO{5, FT}, ::Val{0}, ::Type{Nothing}, args...) where FT = ( FT(1/5),   FT(77/60), -FT(43/60),   FT(17/60), -FT(1/20))
@inline coeff_left_p(scheme::WENO{5, FT}, ::Val{1}, ::Type{Nothing}, args...) where FT = (-FT(1/20),  FT(9/20),   FT(47/60),  -FT(13/60),  FT(1/30))
@inline coeff_left_p(scheme::WENO{5, FT}, ::Val{2}, ::Type{Nothing}, args...) where FT = ( FT(1/30), -FT(13/60),  FT(47/60),   FT(9/20),  -FT(1/20))
@inline coeff_left_p(scheme::WENO{5, FT}, ::Val{3}, ::Type{Nothing}, args...) where FT = (-FT(1/20),  FT(17/60), -FT(43/60),   FT(77/60),  FT(1/5))
@inline coeff_left_p(scheme::WENO{5, FT}, ::Val{4}, ::Type{Nothing}, args...) where FT = ( FT(1/5),  -FT(21/20),  FT(137/60), -FT(163/60), FT(137/60))

@inline coeff_left_p(scheme::WENO{6, FT}, ::Val{0}, ::Type{Nothing}, args...) where FT = ( FT(1/6),   FT(29/20), -FT(21/20),  FT(37/60), -FT(13/60),  FT(1/30))
@inline coeff_left_p(scheme::WENO{6, FT}, ::Val{1}, ::Type{Nothing}, args...) where FT = (-FT(1/30),  FT(11/30),  FT(19/20), -FT(23/60),  FT(7/60),  -FT(1/60))
@inline coeff_left_p(scheme::WENO{6, FT}, ::Val{2}, ::Type{Nothing}, args...) where FT = ( FT(1/60), -FT(2/15),   FT(37/60),  FT(37/60), -FT(2/15),   FT(1/60))
@inline coeff_left_p(scheme::WENO{6, FT}, ::Val{3}, ::Type{Nothing}, args...) where FT = (-FT(1/60),  FT(7/60),  -FT(23/60),  FT(19/20),  FT(11/30), -FT(1/30))
@inline coeff_left_p(scheme::WENO{6, FT}, ::Val{4}, ::Type{Nothing}, args...) where FT = ( FT(1/30), -FT(13/60),  FT(37/60), -FT(21/20),  FT(29/20),  FT(1/6))
@inline coeff_left_p(scheme::WENO{6, FT}, ::Val{5}, ::Type{Nothing}, args...) where FT = (-FT(1/6),   FT(31/30), -FT(163/60), FT(79/20), -FT(71/20),  FT(49/20))

@inline coeff_right_p(scheme::WENO{N}, ::Val{M}, ::Type{Nothing}, args...) where {N, M} = reverse(coeff_left_p₁(scheme, Val(N-M-1), Nothing, args...)) 

@inline  left_biased_p(scheme, M, ψ, args...) = @inbounds  sum(coeff_left_p(scheme, Val(M), args...) .* ψ)
@inline right_biased_p(scheme, M, ψ, args...) = @inbounds sum(coeff_right_p(scheme, Val(M), args...) .* ψ)

# _UNIFORM_ smoothness coefficients (stretched smoothness coefficients are not allowed unless for 5th order WENO)
@inline coeff_β(scheme::WENO{2, FT}, ::Val{0}) where FT = FT.((1, -2, 1))
@inline coeff_β(scheme::WENO{2, FT}, ::Val{1}) where FT = FT.((1, -2, 1))

@inline coeff_β(scheme::WENO{3, FT}, ::Val{0}) where FT = FT.((1, -2, 1))
@inline coeff_β(scheme::WENO{3, FT}, ::Val{1}) where FT = FT.((1, -2, 1))
@inline coeff_β(scheme::WENO{3, FT}, ::Val{2}) where FT = FT.((1, -2, 1))

@inline coeff_β(scheme::WENO{4, FT}, ::Val{0}) where FT = FT.((547,  -3882, 4642, -1854, 7043,  -17246, 7042, 11003, -9402, 2107))
@inline coeff_β(scheme::WENO{4, FT}, ::Val{1}) where FT = FT.((267,  -1642, 1602, -494,  2843,  -5966,  1922, 3443,  -2522,  547))
@inline coeff_β(scheme::WENO{4, FT}, ::Val{2}) where FT = FT.((547,  -2522, 1922, -494,  3443,  -5966,  1602, 2843,  -1642,  267))
@inline coeff_β(scheme::WENO{4, FT}, ::Val{3}) where FT = FT.((2107, -9402, 7042, -1854, 11003, -17246, 4642, 7043,  -3882,  547))

@inline coeff_β(scheme::WENO{5, FT}, ::Val{0}) where FT = FT.((22658,  -208501, 364863, -288007, 86329, 482963,  -1704396, 1358458, -411487, 1521393, -2462076, 758823, 1020563, -649501, 107918))
@inline coeff_β(scheme::WENO{5, FT}, ::Val{1}) where FT = FT.((6908,   -60871,  99213,  -70237,  18079, 138563,  -464976,  337018,  -88297,  406293,  -611976,  165153, 242723,  -140251,  22658))
@inline coeff_β(scheme::WENO{5, FT}, ::Val{2}) where FT = FT.((6908,   -51001,  67923,  -38947,  8209,  104963,  -299076,  179098,  -38947,  231153,  -299076,  67923,  104963,  -51001,    6908))
@inline coeff_β(scheme::WENO{5, FT}, ::Val{3}) where FT = FT.((22658,  -140251, 165153, -88297,  18079, 242723,  -611976,  337018,  -70237,  406293,  -464976,  99213,  138563,  -60871,    6908))
@inline coeff_β(scheme::WENO{5, FT}, ::Val{4}) where FT = FT.((107918, -649501, 758823, -411487, 86329, 1020563, -2462076, 1358458, -288007, 1521393, -1704396, 364863, 482963,  -208501,  22658))

@inline coeff_β(scheme::WENO{6, FT}, ::Val{0}) where FT = FT.((1152561, -12950184, 29442256, -33918804, 19834350, -4712740, 36480687, -166461044, 192596472, -113206788, 27060170, 190757572, -444003904, 262901672, -63394124, 260445372, -311771244, 76206736, 94851237, -47460464, 6150211))
@inline coeff_β(scheme::WENO{6, FT}, ::Val{1}) where FT = FT.((271779,  -3015728,  6694608,  -7408908,  4067018,  -880548,  8449957,  -37913324,  42405032,  -23510468,  5134574,  43093692,  -97838784,  55053752,  -12183636, 56662212,  -65224244,  14742480, 19365967, -9117992,  1152561))
@inline coeff_β(scheme::WENO{6, FT}, ::Val{2}) where FT = FT.((139633,  -1429976,  2863984,  -2792660,  1325006,  -245620,  3824847,  -15880404,  15929912,  -7727988,   1458762,  17195652,  -35817664,  17905032,  -3462252,  19510972,  -20427884,  4086352,  5653317,  -2380800,  271779))
@inline coeff_β(scheme::WENO{6, FT}, ::Val{3}) where FT = FT.((271779,  -2380800,  4086352,  -3462252,  1458762,  -245620,  5653317,  -20427884,  17905032,  -7727988,   1325006,  19510972,  -35817664,  15929912,  -2792660,  17195652,  -15880404,  2863984,  3824847,  -1429976,  139633))
@inline coeff_β(scheme::WENO{6, FT}, ::Val{4}) where FT = FT.((1152561, -9117992,  14742480, -12183636, 5134574,  -880548,  19365967, -65224244,  55053752,  -23510468,  4067018,  56662212,  -97838784,  42405032,  -7408908,  43093692,  -37913324,  6694608,  8449957,  -3015728,  271779))
@inline coeff_β(scheme::WENO{6, FT}, ::Val{5}) where FT = FT.((6150211, -47460464, 76206736, -63394124, 27060170, -4712740, 94851237, -311771244, 262901672, -113206788, 19834350, 260445372, -444003904, 19259647,  -33918804, 190757572, -166461044, 29442256, 36480687, -12950184, 1152561))

# The rule for calculating smoothness indicators is the following (example WENO{4} which is seventh order) 
# ψ[1] (C[1]  * ψ[1] + C[2] * ψ[2] + C[3] * ψ[3] + C[4] * ψ[4]) + 
# ψ[2] (C[5]  * ψ[2] + C[6] * ψ[3] + C[7] * ψ[4]) + 
# ψ[3] (C[8]  * ψ[3] + C[9] * ψ[4])
# ψ[4] (C[10] * ψ[4])

@inline function left_biased_β(ψ, scheme::WENO{N, FT}, stencil) where {N, FT}
    β = FT(0)
    c_idx = 1
    C = coeff_β(scheme, Val(stencil))
    @unroll for idx in 1:N
        β += ψ[idx] * sum(C[c_idx:c_idx+N-1] .* ψ[idx:end])
        c_idx += N
    end
    return β
end

@inline function right_biased_β(ψ, scheme::WENO{N, FT}, stencil) where {N, FT}
    β = FT(0)
    c_idx = 1
    C = coeff_β(scheme, Val(N-stencil-1))
    @unroll for idx in 1:N
        β += ψ[idx] * sum(C[c_idx:c_idx+N-1] .* ψ[idx:end])
        c_idx += N
    end
    return β
end

for (side, coeffs) in zip([:left, :right], ([]))
    biased_weno_weights = Symbol(side, :_biased_weno_weights)
    biased_β = Symbol(side, :_biased_β)
    
    tangential_stencil_u = Symbol(:tangential_, side, :_stencil_u)
    tangential_stencil_v = Symbol(:tangential_, side, :_stencil_v)

    biased_stencil_z = Symbol(side, :_stencil_z)
    
    @eval begin
        @inline function $biased_weno_weights(ψ, scheme::WENO{N, FT}, args...) where {N, FT}

            β = Vector(FT, N)
            α = Vector(FT, N)
            @unroll for idx in 1:N
                β[idx] = $biased_β(ψ[idx], scheme, idx-1)
            end
            
            if scheme isa ZWENO
                τ₅ = abs(β[end] - β[1])
                @unroll for idx in 1:N
                    α[idx] = FT(Cw(scheme, Val(idx-1))) * (1 + (τ₅ / (β[idx] + FT(ε)))^ƞ) 
                end
            else
                @unroll for idx in 1:N
                    α[idx] = FT(Cw(scheme, Val(idx-1))) / (β[idx] + FT(ε))^ƞ
                end
            end
            return α ./ sum(α)
        end

        @inline function $biased_weno_weights(ijk, scheme::WENO{N, FT}, dir, ::Type{VelocityStencil}, u, v) where {N, FT}
            i, j, k = ijk
            
            uₛ = reverse($tangential_stencil_u(i, j, k, dir, u))
            vₛ = reverse($tangential_stencil_v(i, j, k, dir, v))
        
            β = Vector(FT, N)
            α = Vector(FT, N)
            @unroll for idx in 1:N
                β[idx] = 0.5*($biased_β(uₛ[idx], scheme, idx-1) + $biased_β(vₛ[idx], scheme, idx-1))
            end
            
            if scheme isa ZWENO
                τ₅ = abs(β[end] - β[1])
                @unroll for idx in 1:N
                    α[idx] = FT(Cw(scheme, Val(idx-1))) * (1 + (τ₅ / (β[idx] + FT(ε)))^ƞ) 
                end
            else
                @unroll for idx in 1:N
                    α[idx] = FT(Cw(scheme, Val(idx-1))) / (β[idx] + FT(ε))^ƞ
                end
            end
            return α ./ sum(α)
        end

        @inline function $biased_weno_weights(ijk, scheme::WENO{N, FT}, ::Val{3}, ::Type{VelocityStencil}, u) where {N, FT}
            i, j, k = ijk
            
            uₛ = reverse($biased_stencil_z(i, j, k, u))
        
            β₀ = $biased_β₀(FT, u₀, T, scheme, Val(3), idx, loc)
            β₁ = $biased_β₁(FT, u₁, T, scheme, Val(3), idx, loc)
            β₂ = $biased_β₂(FT, u₂, T, scheme, Val(3), idx, loc)
            
            β = Vector(FT, N)
            α = Vector(FT, N)
            @unroll for idx in 1:N
                β[idx] = $biased_β(uₛ[idx], scheme, idx-1)
            end
            
            if scheme isa ZWENO
                τ₅ = abs(β[end] - β[1])
                @unroll for idx in 1:N
                    α[idx] = FT(Cw(scheme, Val(idx-1))) * (1 + (τ₅ / (β[idx] + FT(ε)))^ƞ) 
                end
            else
                @unroll for idx in 1:N
                    α[idx] = FT(Cw(scheme, Val(idx-1))) / (β[idx] + FT(ε))^ƞ
                end
            end
            return α ./ sum(α)
        end
    end
end

for (interp, dir, val, cT) in zip([:xᶠᵃᵃ, :yᵃᶠᵃ, :zᵃᵃᶠ], [:x, :y, :z], [1, 2, 3], [:XT, :YT, :ZT]) 
    for side in (:left, :right)
        interpolate_func = Symbol(:weno_, side, :_biased_interpolate_, interp)
        stencil       = Symbol(side, :_stencil_, dir)
        weno_weights = Symbol(side, :_biased_weno_weights)
        biased_p = Symbol(side, :_biased_p)
        
        @eval begin
            @inline function $interpolate_func(i, j, k, grid, 
                                               scheme::WENO{N, FT, XT, YT, ZT}, 
                                               ψ, idx, loc, args...) where {N, FT, XT, YT, ZT}
                
                ψₜ = reverse($stencil(i, j, k, ψ, grid, args...))
                w = $weno_weights(pass_stencil(ψₜ, i, j, k, Nothing), scheme, Val($val), Nothing, args...)

                ψᵣ = FT(0)
                @unroll for i in 1:N
                    ψᵣ += w[i] * $biased_p(scheme, i, ψₜ[i], $cT, Val($val), idx, loc)
                end
                return ψᵣ
            end

            @inline function $interpolate_func(i, j, k, grid, 
                                               scheme::WENOVectorInvariantN{N, FT, XT, YT, ZT}, 
                                               ψ, idx, loc, VI, args...) where {N, FT, XT, YT, ZT}

                ψₜ = reverse($stencil(i, j, k, ψ, grid, args...))
                w = $weno_weights(pass_stencil(ψₜ, i, j, k, VI), scheme, Val($val), VI, args...)
                ψᵣ = FT(0)
                @unroll for i in 1:N
                    ψᵣ += w[i] * $biased_p(scheme, i, ψₜ[i], $cT, Val($val), idx, loc)
                end
                return ψᵣ
            end
        end
    end
end
