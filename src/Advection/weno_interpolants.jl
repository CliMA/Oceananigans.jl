## Values taken from Balsara & Shu "Monotonicity Preserving Weighted Essentially Non-oscillatory Schemes with Inceasingly High Order of Accuracy"

# Optimal WENO coefficients
@inline Cl(::WENO{2}, ::Val{0}) = 2/3;   @inline Cl(::WENO{2}, ::Val{1}) = 1/3
@inline Cl(::WENO{3}, ::Val{0}) = 3/10;  @inline Cl(::WENO{3}, ::Val{1}) = 3/5;    @inline Cl(::WENO{3}, ::Val{2}) = 1/10
@inline Cl(::WENO{4}, ::Val{0}) = 4/35;  @inline Cl(::WENO{4}, ::Val{1}) = 18/35;  @inline Cl(::WENO{4}, ::Val{2}) = 12/35;   @inline Cl(::WENO{4}, ::Val{3}) = 1/35;  
@inline Cl(::WENO{5}, ::Val{0}) = 5/126; @inline Cl(::WENO{5}, ::Val{1}) = 20/63;  @inline Cl(::WENO{5}, ::Val{2}) = 10/21;   @inline Cl(::WENO{5}, ::Val{3}) = 10/63; @inline Cl(::WENO{5}, ::Val{4}) = 1/126; 
@inline Cl(::WENO{6}, ::Val{0}) = 1/77;  @inline Cl(::WENO{6}, ::Val{1}) = 25/154; @inline Cl(::WENO{6}, ::Val{2}) = 100/231; @inline Cl(::WENO{6}, ::Val{3}) = 25/77; @inline Cl(::WENO{6}, ::Val{4}) = 5/77;  @inline Cl(::WENO{6}, ::Val{5}) = 1/462;

# ENO reconstruction procedure per stencil 
for buffer in [2, 3, 4, 5, 6]
    for stencil in collect(0:1:buffer-1)

        # ENO coefficients for uniform direction (::Type{Nothing}) and stretched directions (T) directions 
        @eval begin
            @inline Cr(scheme::WENO{$buffer}, ::Val{$stencil}) = Cl(scheme, Val($(buffer-stencil-1)))

            # uniform coefficients are independent on direction and location
            @inline  coeff_left_p(scheme::WENO{$buffer, FT}, ::Val{$stencil}, ::Type{Nothing}, args...) where FT = FT.($(stencil_coefficients(50, stencil  , collect(1:100), collect(1:100); order = buffer)))
            @inline coeff_right_p(scheme::WENO{$buffer, FT}, ::Val{$stencil}, ::Type{Nothing}, args...) where FT = FT.($(stencil_coefficients(50, stencil-1, collect(1:100), collect(1:100); order = buffer)))

            # stretched coefficients are precalculated
            @inline  coeff_left_p(scheme::WENO{$buffer}, ::Val{$stencil}, T, dir, i, loc) = retrieve_coeff(scheme, $stencil,     dir, i, loc)
            @inline coeff_right_p(scheme::WENO{$buffer}, ::Val{$stencil}, T, dir, i, loc) = retrieve_coeff(scheme, $(stencil-1), dir, i, loc)
        end
    
        # left biased and right biased reconstruction value for each stencil
        @eval begin
            @inline  left_biased_p(scheme::WENO{$buffer}, ::Val{$stencil}, ψ, T, dir, i, loc) =  sum(coeff_left_p(scheme, Val($stencil), T, dir, i, loc) .* ψ)
            @inline right_biased_p(scheme::WENO{$buffer}, ::Val{$stencil}, ψ, T, dir, i, loc) = sum(coeff_right_p(scheme, Val($stencil), T, dir, i, loc) .* ψ)
        end
    end
end

# _UNIFORM_ smoothness coefficients (stretched smoothness coefficients are to be fixed!)
@inline coeff_β(scheme::WENO{2, FT}, ::Val{0}) where FT = FT.((1, -2, 1))
@inline coeff_β(scheme::WENO{2, FT}, ::Val{1}) where FT = FT.((1, -2, 1))

@inline coeff_β(scheme::WENO{3, FT}, ::Val{0}) where FT = FT.((10, -31, 11, 25, -19, 4))
@inline coeff_β(scheme::WENO{3, FT}, ::Val{1}) where FT = FT.((4,  -13, 5,  13, -13, 4))
@inline coeff_β(scheme::WENO{3, FT}, ::Val{2}) where FT = FT.((4,  -19, 11, 25, -31, 10))

@inline coeff_β(scheme::WENO{4, FT}, ::Val{0}) where FT = FT.((2107, -9402, 7042, -1854, 11003, -17246, 4642, 7043,  -3882,  547))
@inline coeff_β(scheme::WENO{4, FT}, ::Val{1}) where FT = FT.((547,  -2522, 1922, -494,  3443,  -5966,  1602, 2843,  -1642,  267))
@inline coeff_β(scheme::WENO{4, FT}, ::Val{2}) where FT = FT.((267,  -1642, 1602, -494,  2843,  -5966,  1922, 3443,  -2522,  547))
@inline coeff_β(scheme::WENO{4, FT}, ::Val{3}) where FT = FT.((547,  -3882, 4642, -1854, 7043,  -17246, 7042, 11003, -9402, 2107))

@inline coeff_β(scheme::WENO{5, FT}, ::Val{0}) where FT = FT.((107918, -649501, 758823, -411487, 86329, 1020563, -2462076, 1358458, -288007, 1521393, -1704396, 364863, 482963,  -208501,  22658))
@inline coeff_β(scheme::WENO{5, FT}, ::Val{1}) where FT = FT.((22658,  -140251, 165153, -88297,  18079, 242723,  -611976,  337018,  -70237,  406293,  -464976,  99213,  138563,  -60871,    6908))
@inline coeff_β(scheme::WENO{5, FT}, ::Val{2}) where FT = FT.((6908,   -51001,  67923,  -38947,  8209,  104963,  -299076,  179098,  -38947,  231153,  -299076,  67923,  104963,  -51001,    6908))
@inline coeff_β(scheme::WENO{5, FT}, ::Val{3}) where FT = FT.((6908,   -60871,  99213,  -70237,  18079, 138563,  -464976,  337018,  -88297,  406293,  -611976,  165153, 242723,  -140251,  22658))
@inline coeff_β(scheme::WENO{5, FT}, ::Val{4}) where FT = FT.((22658,  -208501, 364863, -288007, 86329, 482963,  -1704396, 1358458, -411487, 1521393, -2462076, 758823, 1020563, -649501, 107918))

@inline coeff_β(scheme::WENO{6, FT}, ::Val{0}) where FT = FT.((6150211, -47460464, 76206736, -63394124, 27060170, -4712740, 94851237, -311771244, 262901672, -113206788, 19834350, 260445372, -444003904, 192596472, -33918804, 190757572, -166461044, 29442256, 36480687, -12950184, 1152561))
@inline coeff_β(scheme::WENO{6, FT}, ::Val{1}) where FT = FT.((1152561, -9117992,  14742480, -12183636, 5134574,  -880548,  19365967, -65224244,  55053752,  -23510468,  4067018,  56662212,  -97838784,  42405032,  -7408908,  43093692,  -37913324,  6694608,  8449957,  -3015728,  271779))
@inline coeff_β(scheme::WENO{6, FT}, ::Val{2}) where FT = FT.((271779,  -2380800,  4086352,  -3462252,  1458762,  -245620,  5653317,  -20427884,  17905032,  -7727988,   1325006,  19510972,  -35817664,  15929912,  -2792660,  17195652,  -15880404,  2863984,  3824847,  -1429976,  139633))
@inline coeff_β(scheme::WENO{6, FT}, ::Val{3}) where FT = FT.((139633,  -1429976,  2863984,  -2792660,  1325006,  -245620,  3824847,  -15880404,  15929912,  -7727988,   1458762,  17195652,  -35817664,  17905032,  -3462252,  19510972,  -20427884,  4086352,  5653317,  -2380800,  271779))
@inline coeff_β(scheme::WENO{6, FT}, ::Val{4}) where FT = FT.((271779,  -3015728,  6694608,  -7408908,  4067018,  -880548,  8449957,  -37913324,  42405032,  -23510468,  5134574,  43093692,  -97838784,  55053752,  -12183636, 56662212,  -65224244,  14742480, 19365967, -9117992,  1152561))
@inline coeff_β(scheme::WENO{6, FT}, ::Val{5}) where FT = FT.((1152561, -12950184, 29442256, -33918804, 19834350, -4712740, 36480687, -166461044, 192596472, -113206788, 27060170, 190757572, -444003904, 262901672, -63394124, 260445372, -311771244, 76206736, 94851237, -47460464, 6150211))

# The rule for calculating smoothness indicators is the following (example WENO{4} which is seventh order) 
# ψ[1] (C[1]  * ψ[1] + C[2] * ψ[2] + C[3] * ψ[3] + C[4] * ψ[4]) + 
# ψ[2] (C[5]  * ψ[2] + C[6] * ψ[3] + C[7] * ψ[4]) + 
# ψ[3] (C[8]  * ψ[3] + C[9] * ψ[4])
# ψ[4] (C[10] * ψ[4])
# This expression is the output of metaprogrammed_smoothness_sum(4)

# Trick to force compilation of Val(stencil-1) and avoid loops on the GPU
function metaprogrammed_smoothness_sum(buffer)
    elem = Vector(undef, buffer)
    c_idx = 1
    for stencil = 1:buffer - 1
        stencil_sum   = Expr(:call, :+, (:(C[$(c_idx + i - stencil)] * ψ[$i]) for i in stencil:buffer)...)
        elem[stencil] = :(ψ[$stencil] * $stencil_sum)
        c_idx += buffer - stencil + 1
    end

    elem[buffer] = :(ψ[$buffer] * ψ[$buffer] * C[$c_idx])
    
    return Expr(:call, :+, elem...)
end

for buffer in [2, 3, 4, 5, 6]
    @eval begin
        @inline smoothness_sum(scheme::WENO{$buffer}, ψ, C) = @inbounds $(metaprogrammed_smoothness_sum(buffer))
    end
end

for buffer in [2, 3, 4, 5, 6], stencil in [0, 1, 2, 3, 4, 5]
    @eval begin
        @inline left_biased_β(ψ, scheme::WENO{$buffer, FT}, ::Val{$stencil})  where {FT} = @inbounds smoothness_sum(scheme, ψ, coeff_β(scheme, Val($stencil)))
        @inline right_biased_β(ψ, scheme::WENO{$buffer, FT}, ::Val{$stencil}) where {FT} = @inbounds smoothness_sum(scheme, reverse(ψ), coeff_β(scheme, Val($(buffer-stencil-1))))
    end
end

function metaprogrammed_beta_loop(buffer)
    elem = Vector(undef, buffer)
    for stencil = 1:buffer
        elem[stencil] = :(func(ψ[$stencil], scheme, Val($(stencil-1))))
    end

    return :($(elem...), )
end

function metaprogrammed_zweno_alpha_loop(buffer)
    elem = Vector(undef, buffer)
    for stencil = 1:buffer
        elem[stencil] = :(FT(coeff(scheme, Val($(stencil-1)))) * (1 + (τ / (β[$stencil] + FT(ε)))^ƞ))
    end

    return :($(elem...), )
end

function metaprogrammed_js_alpha_loop(buffer)
    elem = Vector(undef, buffer)
    for stencil = 1:buffer
        elem[stencil] = :(FT(coeff(scheme, Val($(stencil-1)))) / (β[$stencil] + FT(ε))^ƞ)
    end

    return :($(elem...), )
end

for buffer in [2, 3, 4, 5, 6]
    @eval begin
        @inline        beta_loop(scheme::WENO{$buffer}, ψ, func)          = @inbounds $(metaprogrammed_beta_loop(buffer))
        @inline zweno_alpha_loop(scheme::WENO{$buffer}, β, τ, coeff, FT)  = @inbounds $(metaprogrammed_zweno_alpha_loop(buffer))
        @inline    js_alpha_loop(scheme::WENO{$buffer}, β, coeff, FT)     = @inbounds $(metaprogrammed_js_alpha_loop(buffer))
    end
end

@inline global_smoothness_indicator(::Val{2}, β) = nothing
@inline global_smoothness_indicator(::Val{3}, β) = abs(β[1] - β[end])
@inline global_smoothness_indicator(::Val{4}, β) = abs(β[1] - β[2] - β[3] + β[4])
@inline global_smoothness_indicator(::Val{5}, β) = abs(β[1] - β[end])
@inline global_smoothness_indicator(::Val{6}, β) = abs(β[1] - β[2] - β[5] + β[6])

# Calculating Dynamic WENO Weights, either with JS weno, Z weno or VectorInvariant WENO
for (side, coeff) in zip([:left, :right], (:Cl, :Cr))
    biased_weno_weights = Symbol(side, :_biased_weno_weights)
    biased_β = Symbol(side, :_biased_β)
    
    tangential_stencil_u = Symbol(:tangential_, side, :_stencil_u)
    tangential_stencil_v = Symbol(:tangential_, side, :_stencil_v)

    biased_stencil_z = Symbol(side, :_stencil_z)
    
    @eval begin
        @inline function $biased_weno_weights(ψ, scheme::WENO{N, FT}, args...) where {N, FT}
            β = beta_loop(scheme, ψ, $biased_β)
                
            if scheme isa ZWENO
                τ = global_smoothness_indicator(Val(N), β)
                α = zweno_alpha_loop(scheme, β, τ, $coeff, FT)
            else
                α = js_alpha_loop(scheme, β, $coeff, FT)
            end
            return α ./ sum(α)
        end

        @inline function $biased_weno_weights(ijk, scheme::WENO{N, FT}, dir, ::Type{VelocityStencil}, u, v) where {N, FT}
            i, j, k = ijk
            
            uₛ = $tangential_stencil_u(i, j, k, scheme, dir, u)
            vₛ = $tangential_stencil_v(i, j, k, scheme, dir, v)
        
            βᵤ = beta_loop(scheme, uₛ, $biased_β)
            βᵥ = beta_loop(scheme, vₛ, $biased_β)

            β  = βᵤ .+ βᵥ
            
            if scheme isa ZWENO
                τ = global_smoothness_indicator(Val(N), β)
                α = zweno_alpha_loop(scheme, β, τ, $coeff, FT)
            else
                α  = js_alpha_loop(scheme, β, $coeff, FT)
            end
            return α ./ sum(α)
        end

        @inline function $biased_weno_weights(ijk, scheme::WENO{N, FT}, ::Val{3}, ::Type{VelocityStencil}, u) where {N, FT}
            i, j, k = ijk
            
            uₛ = $biased_stencil_z(i, j, k, scheme, u)
        
            β = beta_loop(scheme, uₛ, $biased_β)
            
            if scheme isa ZWENO
                τ = global_smoothness_indicator(Val(N), β)
                α = zweno_alpha_loop(scheme, β, τ, $coeff, FT)
            else
                α  = js_alpha_loop(scheme, β, $coeff, FT)
            end
            return α ./ sum(α)
        end
    end
end

function calc_weno_stencil(buffer, shift, dir, func) 
    N = buffer * 2
    if shift != :none
        N -=1
    end
    stencil_full = Vector(undef, buffer)
    rng = 1:N
    if shift == :right
        rng = rng .+ 1
    end
    for stencil in 1:buffer
        stencil_point = Vector(undef, buffer)
        rngstencil = rng[stencil:stencil+buffer-1]
        for (idx, n) in enumerate(rngstencil)
            c = n - buffer - 1
            if func 
                stencil_point[idx] =  dir == :x ? 
                                    :(ψ(i + $c, j, k, args...)) :
                                    dir == :y ?
                                    :(ψ(i, j + $c, k, args...)) :
                                    :(ψ(i, j, k + $c, args...))
            else    
                stencil_point[idx] =  dir == :x ? 
                                    :(ψ[i + $c, j, k]) :
                                    dir == :y ?
                                    :(ψ[i, j + $c, k]) :
                                    :(ψ[i, j, k + $c])
            end                
        end
        stencil_full[buffer - stencil + 1] = :($(stencil_point...), )
    end
    return :($(stencil_full...), )
end

for side in (:left, :right), dir in (:x, :y, :z)
    stencil = Symbol(side, :_stencil_, dir)

    for buffer in [2, 3, 4, 5, 6]
        @eval begin
            @inline $stencil(i, j, k, scheme::WENO{$buffer}, ψ, args...)           = @inbounds $(calc_weno_stencil(buffer, side, dir, false))
            @inline $stencil(i, j, k, scheme::WENO{$buffer}, ψ::Function, args...) = @inbounds $(calc_weno_stencil(buffer, side, dir,  true))
        end
    end
end

# Stencil for vector invariant calculation of smoothness indicators in the horizontal direction
# Parallel to the interpolation direction! (same as left/right stencil)
for (vel, interp) in zip([:u, :v], [:ℑyᵃᶠᵃ, :ℑxᶠᵃᵃ]), side in [:left, :right], (dir, ξ) in zip([1, 2], (:x, :y))
    tangential_stencil = Symbol(:tangential_, side, :_stencil_, vel)
    biased_stencil = Symbol(side, :_stencil_, ξ)
    @eval begin
        @inline $tangential_stencil(i, j, k, scheme::WENO, ::Val{$dir}, $vel) = @inbounds $biased_stencil(i, j, k, scheme, $interp, $vel)
    end
end

# Trick to force compilation of Val(stencil-1) and avoid loops on the GPU
function metaprogrammed_stencil_sum(buffer)
    elem = Vector(undef, buffer)
    for stencil = 1:buffer
        elem[stencil] = :(w[$stencil] * func(scheme, Val($(stencil-1)), ψ[$stencil], cT, Val(val), idx, loc))
    end

    return Expr(:call, :+, elem...)
end

for buffer in [2, 3, 4, 5, 6]
    @eval begin
        @inline stencil_sum(scheme::WENO{$buffer}, ψ, w, func, cT, val, idx, loc) = @inbounds $(metaprogrammed_stencil_sum(buffer))
    end
end

# Interpolation functions
for (interp, dir, val, cT) in zip([:xᶠᵃᵃ, :yᵃᶠᵃ, :zᵃᵃᶠ], [:x, :y, :z], [1, 2, 3], [:XT, :YT, :ZT]) 
    for side in (:left, :right)
        interpolate_func = Symbol(:stretched_, side, :_biased_interpolate_, interp)
        stencil       = Symbol(side, :_stencil_, dir)
        weno_weights = Symbol(side, :_biased_weno_weights)
        biased_p = Symbol(side, :_biased_p)
        
        @eval begin
            @inline function $interpolate_func(i, j, k, grid, 
                                               scheme::WENO{N, FT, XT, YT, ZT}, 
                                               ψ, idx, loc, args...) where {N, FT, XT, YT, ZT}
                ψₜ = $stencil(i, j, k, scheme, ψ, grid, args...)
                w = $weno_weights(ψₜ, scheme, Val($val), Nothing, args...)
                return stencil_sum(scheme, ψₜ, w, $biased_p, $cT, $val, idx, loc)
            end

            @inline function $interpolate_func(i, j, k, grid, 
                                               scheme::WENOVectorInvariant{N, FT, XT, YT, ZT}, 
                                               ψ, idx, loc, VI::Type{VorticityStencil}, args...) where {N, FT, XT, YT, ZT}

                ψₜ = $stencil(i, j, k, scheme, ψ, grid, args...)
                w = $weno_weights(ψₜ, scheme, Val($val), VI, args...)
                return stencil_sum(scheme, ψₜ, w, $biased_p, $cT, $val, idx, loc)
            end

            @inline function $interpolate_func(i, j, k, grid, 
                                               scheme::WENOVectorInvariant{N, FT, XT, YT, ZT}, 
                                               ψ, idx, loc, VI::Type{VelocityStencil}, args...) where {N, FT, XT, YT, ZT}

                ψₜ = $stencil(i, j, k, scheme, ψ, grid, args...)
                w = $weno_weights((i, j, k), scheme, Val($val), VI, args...)
                return stencil_sum(scheme, ψₜ, w, $biased_p, $cT, $val, idx, loc)
            end
        end
    end
end
