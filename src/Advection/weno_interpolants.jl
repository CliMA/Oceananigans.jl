using Oceananigans.Operators: ℑyᵃᶠᵃ, ℑxᶠᵃᵃ

# WENO reconstruction of order `M` entails reconstructions of order `N`
# on `N` different stencils, where `N = (M + 1) / 2`.
#
# Each reconstruction `r` at cell `i` is denoted
# 
# `v̂ᵢᵣ = ∑ⱼ(cᵣⱼ v̅ᵢ₋ᵣ₊ⱼ)` 
# 
# where j ranges from 0 to N and the coefficients cᵣⱼ for each stencil r
# are given by `coeff_side_p(scheme, Val(r))`.
# 
# The different reconstructions are combined to provide a
# "higher-order essentially non-oscillatory" reconstruction,
# 
# `v⋆ᵢ = ∑ᵣ(wᵣ v̂ᵣ)`
# 
# where the weights wᵣ are calculated dynamically with `side_biased_weno_weights(ψ, scheme)`.
#

""" 
`AbstractSmoothnessStencil`s specifies the polynomials used for diagnosing stencils' smoothness for weno weights 
calculation in the `VectorInvariant` advection formulation. 

Smoothness polynomials different from reconstructing polynomials can be specified _only_ for functional reconstructions:
```julia
_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, reconstruced_function::F, smoothness_stencil, args...) where F<:Function
```

For scalar reconstructions 
```julia
_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, reconstruced_field::F) where F<:AbstractField
```
the smoothness is _always_ diagnosed from the reconstructing polynomials of `reconstructed_field`

Options:
========

- `DefaultStencil`: uses the same polynomials used for reconstruction
- `VelocityStencil`: is valid _only_ for vorticity reconstruction and diagnoses the smoothness based on 
                     `(Face, Face, Center)` polynomial interpolations of `u` and `v`
- `FunctionStencil`: allows using a custom function as smoothness indicator. 
The custom function should share arguments with the reconstructed function. 

Example:
========

```julia
@inline   smoothness_function(i, j, k, grid, args...) = custom_smoothness_function(i, j, k, grid, args...)
@inline reconstruced_function(i, j, k, grid, args...) = custom_reconstruction_function(i, j, k, grid, args...)

smoothness_stencil = FunctionStencil(smoothness_function)    
```
"""
abstract type AbstractSmoothnessStencil end

"""`DefaultStencil <: AbstractSmoothnessStencil`, see `AbstractSmoothnessStencil`"""
struct DefaultStencil <:AbstractSmoothnessStencil end

"""`VelocityStencil <: AbstractSmoothnessStencil`, see `AbstractSmoothnessStencil`"""
struct VelocityStencil <:AbstractSmoothnessStencil end

"""`FunctionStencil <: AbstractSmoothnessStencil`, see `AbstractSmoothnessStencil`"""
struct FunctionStencil{F} <:AbstractSmoothnessStencil 
    func :: F
end

Base.show(io::IO, a::FunctionStencil) =  print(io, "FunctionStencil f = $(a.func)")

const ƞ = Int32(2) # WENO exponent
const ε = 1e-8

# Optimal values taken from
# Balsara & Shu, "Monotonicity Preserving Weighted Essentially Non-oscillatory Schemes with Inceasingly High Order of Accuracy"
@inline Cl(::WENO{2}, ::Val{0}) = 2/3
@inline Cl(::WENO{2}, ::Val{1}) = 1/3

@inline Cl(::WENO{3}, ::Val{0}) = 3/10
@inline Cl(::WENO{3}, ::Val{1}) = 3/5
@inline Cl(::WENO{3}, ::Val{2}) = 1/10

@inline Cl(::WENO{4}, ::Val{0}) = 4/35
@inline Cl(::WENO{4}, ::Val{1}) = 18/35
@inline Cl(::WENO{4}, ::Val{2}) = 12/35
@inline Cl(::WENO{4}, ::Val{3}) = 1/35

@inline Cl(::WENO{5}, ::Val{0}) = 5/126
@inline Cl(::WENO{5}, ::Val{1}) = 20/63
@inline Cl(::WENO{5}, ::Val{2}) = 10/21
@inline Cl(::WENO{5}, ::Val{3}) = 10/63
@inline Cl(::WENO{5}, ::Val{4}) = 1/126

@inline Cl(::WENO{6}, ::Val{0}) = 1/77
@inline Cl(::WENO{6}, ::Val{1}) = 25/154
@inline Cl(::WENO{6}, ::Val{2}) = 100/231
@inline Cl(::WENO{6}, ::Val{3}) = 25/77
@inline Cl(::WENO{6}, ::Val{4}) = 5/77
@inline Cl(::WENO{6}, ::Val{5}) = 1/462

# ENO reconstruction procedure per stencil 
for buffer in [2, 3, 4, 5, 6]
    for stencil in collect(0:1:buffer-1)

        # ENO coefficients for uniform direction (when T<:Nothing) and stretched directions (when T<:Any) 
        @eval begin
            # uniform coefficients are independent on direction and location
            @inline coeff_p(::WENO{$buffer, FT}, bias,  ::Val{$stencil}, ::Type{Nothing}, args...) where FT = 
                @inbounds FT.($(stencil_coefficients(50, stencil, collect(1:100), collect(1:100); order = buffer)))

            # stretched coefficients are retrieved from precalculated coefficients
            @inline coeff_p(scheme::WENO{$buffer}, bias,  ::Val{$stencil}, T, dir, i, loc) = 
                ifelse(bias == LeftBias(), retrieve_coeff(scheme, $stencil, dir, i, loc),
                                   reverse(retrieve_coeff(scheme, $(buffer - 2 - stencil), dir, i, loc)))
        end
    
        # left biased and right biased reconstruction value for each stencil
        @eval begin
            @inline biased_p(scheme::WENO{$buffer}, bias, ::Val{$stencil}, ψ, T, dir, i, loc) = 
                @inbounds sum(coeff_p(scheme, bias, Val($stencil), T, dir, i, loc) .* ψ)
        end
    end
end

# _UNIFORM_ smoothness coefficients (stretched smoothness coefficients are to be fixed!)
@inline smoothness_coefficients(scheme::WENO{2, FT}, ::Val{0}) where FT = @inbounds FT.((1, -2, 1))
@inline smoothness_coefficients(scheme::WENO{2, FT}, ::Val{1}) where FT = @inbounds FT.((1, -2, 1))

@inline smoothness_coefficients(scheme::WENO{3, FT}, ::Val{0}) where FT = @inbounds FT.((10, -31, 11, 25, -19,  4))
@inline smoothness_coefficients(scheme::WENO{3, FT}, ::Val{1}) where FT = @inbounds FT.((4,  -13, 5,  13, -13,  4))
@inline smoothness_coefficients(scheme::WENO{3, FT}, ::Val{2}) where FT = @inbounds FT.((4,  -19, 11, 25, -31, 10))

@inline smoothness_coefficients(scheme::WENO{4, FT}, ::Val{0}) where FT = @inbounds FT.((2.107,  -9.402, 7.042, -1.854, 11.003,  -17.246,  4.642,  7.043,  -3.882, 0.547))
@inline smoothness_coefficients(scheme::WENO{4, FT}, ::Val{1}) where FT = @inbounds FT.((0.547,  -2.522, 1.922, -0.494,  3.443,  - 5.966,  1.602,  2.843,  -1.642, 0.267))
@inline smoothness_coefficients(scheme::WENO{4, FT}, ::Val{2}) where FT = @inbounds FT.((0.267,  -1.642, 1.602, -0.494,  2.843,  - 5.966,  1.922,  3.443,  -2.522, 0.547))
@inline smoothness_coefficients(scheme::WENO{4, FT}, ::Val{3}) where FT = @inbounds FT.((0.547,  -3.882, 4.642, -1.854,  7.043,  -17.246,  7.042, 11.003,  -9.402, 2.107))

@inline smoothness_coefficients(scheme::WENO{5, FT}, ::Val{0}) where FT = @inbounds FT.((1.07918,  -6.49501, 7.58823, -4.11487,  0.86329,  10.20563, -24.62076, 13.58458, -2.88007, 15.21393, -17.04396, 3.64863,  4.82963, -2.08501, 0.22658)) 
@inline smoothness_coefficients(scheme::WENO{5, FT}, ::Val{1}) where FT = @inbounds FT.((0.22658,  -1.40251, 1.65153, -0.88297,  0.18079,   2.42723,  -6.11976,  3.37018, -0.70237,  4.06293,  -4.64976, 0.99213,  1.38563, -0.60871, 0.06908)) 
@inline smoothness_coefficients(scheme::WENO{5, FT}, ::Val{2}) where FT = @inbounds FT.((0.06908,  -0.51001, 0.67923, -0.38947,  0.08209,   1.04963,  -2.99076,  1.79098, -0.38947,  2.31153,  -2.99076, 0.67923,  1.04963, -0.51001, 0.06908)) 
@inline smoothness_coefficients(scheme::WENO{5, FT}, ::Val{3}) where FT = @inbounds FT.((0.06908,  -0.60871, 0.99213, -0.70237,  0.18079,   1.38563,  -4.64976,  3.37018, -0.88297,  4.06293,  -6.11976, 1.65153,  2.42723, -1.40251, 0.22658)) 
@inline smoothness_coefficients(scheme::WENO{5, FT}, ::Val{4}) where FT = @inbounds FT.((0.22658,  -2.08501, 3.64863, -2.88007,  0.86329,   4.82963, -17.04396, 13.58458, -4.11487, 15.21393, -24.62076, 7.58823, 10.20563, -6.49501, 1.07918)) 

@inline smoothness_coefficients(scheme::WENO{6, FT}, ::Val{0}) where FT = @inbounds FT.((0.6150211, -4.7460464, 7.6206736, -6.3394124, 2.7060170, -0.4712740,  9.4851237, -31.1771244, 26.2901672, -11.3206788,  1.9834350, 26.0445372, -44.4003904, 19.2596472, -3.3918804, 19.0757572, -16.6461044, 2.9442256, 3.6480687, -1.2950184, 0.1152561)) 
@inline smoothness_coefficients(scheme::WENO{6, FT}, ::Val{1}) where FT = @inbounds FT.((0.1152561, -0.9117992, 1.4742480, -1.2183636, 0.5134574, -0.0880548,  1.9365967,  -6.5224244,  5.5053752,  -2.3510468,  0.4067018,  5.6662212,  -9.7838784,  4.2405032, -0.7408908,  4.3093692,  -3.7913324, 0.6694608, 0.8449957, -0.3015728, 0.0271779)) 
@inline smoothness_coefficients(scheme::WENO{6, FT}, ::Val{2}) where FT = @inbounds FT.((0.0271779, -0.2380800, 0.4086352, -0.3462252, 0.1458762, -0.0245620,  0.5653317,  -2.0427884,  1.7905032,  -0.7727988,  0.1325006,  1.9510972,  -3.5817664,  1.5929912, -0.2792660,  1.7195652,  -1.5880404, 0.2863984, 0.3824847, -0.1429976, 0.0139633)) 
@inline smoothness_coefficients(scheme::WENO{6, FT}, ::Val{3}) where FT = @inbounds FT.((0.0139633, -0.1429976, 0.2863984, -0.2792660, 0.1325006, -0.0245620,  0.3824847,  -1.5880404,  1.5929912,  -0.7727988,  0.1458762,  1.7195652,  -3.5817664,  1.7905032, -0.3462252,  1.9510972,  -2.0427884, 0.4086352, 0.5653317, -0.2380800, 0.0271779)) 
@inline smoothness_coefficients(scheme::WENO{6, FT}, ::Val{4}) where FT = @inbounds FT.((0.0271779, -0.3015728, 0.6694608, -0.7408908, 0.4067018, -0.0880548,  0.8449957,  -3.7913324,  4.2405032,  -2.3510468,  0.5134574,  4.3093692,  -9.7838784,  5.5053752, -1.2183636,  5.6662212,  -6.5224244, 1.4742480, 1.9365967, -0.9117992, 0.1152561)) 
@inline smoothness_coefficients(scheme::WENO{6, FT}, ::Val{5}) where FT = @inbounds FT.((0.1152561, -1.2950184, 2.9442256, -3.3918804, 1.9834350, -0.4712740,  3.6480687, -16.6461044, 19.2596472, -11.3206788,  2.7060170, 19.0757572, -44.4003904, 26.2901672, -6.3394124, 26.0445372, -31.1771244, 7.6206736, 9.4851237, -4.7460464, 0.6150211)) 

# The rule for calculating smoothness indicators is the following (example WENO{4} which is seventh order) 
# ψ[1] (C[1]  * ψ[1] + C[2] * ψ[2] + C[3] * ψ[3] + C[4] * ψ[4]) + 
# ψ[2] (C[5]  * ψ[2] + C[6] * ψ[3] + C[7] * ψ[4]) + 
# ψ[3] (C[8]  * ψ[3] + C[9] * ψ[4])
# ψ[4] (C[10] * ψ[4])
# This expression is the output of metaprogrammed_smoothness_sum(4)

# Trick to force compilation of Val(stencil-1) and avoid loops on the GPU
@inline function metaprogrammed_smoothness_sum(buffer)
    elem = Vector(undef, buffer)
    c_idx = 1
    for stencil = 1:buffer - 1
        stencil_sum   = Expr(:call, :+, (:(@inbounds C[$(c_idx + i - stencil)] * ψ[$i]) for i in stencil:buffer)...)
        elem[stencil] = :(@inbounds ψ[$stencil] * $stencil_sum)
        c_idx += buffer - stencil + 1
    end

    elem[buffer] = :(@inbounds ψ[$buffer] * ψ[$buffer] * C[$c_idx])
    
    return Expr(:call, :+, elem...)
end

# Smoothness indicators for stencil `stencil` for left and right biased reconstruction
for buffer in [2, 3, 4, 5, 6]
    @eval begin
        @inline smoothness_sum(scheme::WENO{$buffer}, ψ, C) = @inbounds $(metaprogrammed_smoothness_sum(buffer))
    end

    for stencil in [0, 1, 2, 3, 4, 5]
        @eval begin
            @inline biased_β(ψ, scheme::WENO{$buffer}, ::Val{$stencil}) = @inbounds smoothness_sum(scheme, ψ, smoothness_coefficients(scheme, Val($stencil)))
        end
    end
end

# Shenanigans for WENO weights calculation for vector invariant formulation -> [β[i] = 0.5*(βᵤ[i] + βᵥ[i]) for i in 1:buffer]
@inline function metaprogrammed_beta_sum(buffer)
    elem = Vector(undef, buffer)
    for stencil = 1:buffer
        elem[stencil] = :(@inbounds (β₁[$stencil] + β₂[$stencil])/2)
    end

    return :($(elem...),)
end

# left and right biased_β calculation for scheme and stencil = 0:buffer - 1
@inline function metaprogrammed_beta_loop(buffer)
    elem = Vector(undef, buffer)
    for stencil = 1:buffer
        elem[stencil] = :(@inbounds func(ψ[$stencil], scheme, Val($(stencil-1))))
    end

    return :($(elem...),)
end

# ZWENO α weights dᵣ * (1 + (τ₂ᵣ₋₁ / (βᵣ + ε))ᵖ)
@inline function metaprogrammed_zweno_alpha_loop(buffer)
    elem = Vector(undef, buffer)
    for stencil = 1:buffer
        elem[stencil] = :(@inbounds FT(coeff(scheme, Val($(stencil-1)))) * (1 + (τ / (β[$stencil] + FT(ε)))^ƞ))
    end

    return :($(elem...),)
end

for buffer in [2, 3, 4, 5, 6]
    @eval begin
        @inline         beta_sum(scheme::WENO{$buffer}, β₁, β₂)          = @inbounds $(metaprogrammed_beta_sum(buffer))
        @inline        beta_loop(scheme::WENO{$buffer}, ψ, func)         = @inbounds $(metaprogrammed_beta_loop(buffer))
        @inline zweno_alpha_loop(scheme::WENO{$buffer}, β, τ, coeff, FT) = @inbounds $(metaprogrammed_zweno_alpha_loop(buffer))
    end
end

# Global smoothness indicator τ₂ᵣ₋₁ taken from "Accuracy of the weighted essentially non-oscillatory conservative finite difference schemes", Don & Borges, 2013
@inline global_smoothness_indicator(::Val{2}, β) = @inbounds abs(β[1] - β[2])
@inline global_smoothness_indicator(::Val{3}, β) = @inbounds abs(β[1] - β[3])
@inline global_smoothness_indicator(::Val{4}, β) = @inbounds abs(β[1] +  3β[2] -   3β[3] -    β[4])
@inline global_smoothness_indicator(::Val{5}, β) = @inbounds abs(β[1] +  2β[2] -   6β[3] +   2β[4] + β[5])
@inline global_smoothness_indicator(::Val{6}, β) = @inbounds abs(β[1] + 36β[2] + 135β[3] - 135β[4] - 36β[5] - β[6])

# Calculating Dynamic WENO Weights (wᵣ), using the Z-WENO formulation
@inline function biased_weno_weights(ψ, scheme::WENO{N, FT}, args...) where {N, FT}
    @inbounds begin
        β = beta_loop(scheme, ψ, biased_β)
                
        τ = global_smoothness_indicator(Val(N), β)
        α = zweno_alpha_loop(scheme, β, τ, Cl, FT)

        return α ./ sum(α)
    end
end

@inline function biased_weno_weights(ijk, scheme::WENO{N, FT}, bias, dir, ::VelocityStencil, u, v) where {N, FT}
    @inbounds begin
        i, j, k = ijk
    
        uₛ = tangential_stencil_u(i, j, k, scheme, bias, dir, u)
        vₛ = tangential_stencil_v(i, j, k, scheme, bias, dir, v)
        βᵤ = beta_loop(scheme, uₛ, biased_β)
        βᵥ = beta_loop(scheme, vₛ, biased_β)
        β  =  beta_sum(scheme, βᵤ, βᵥ)

        τ = global_smoothness_indicator(Val(N), β)
        α = zweno_alpha_loop(scheme, β, τ, Cl, FT)
    
        return α ./ sum(α)
    end
end

""" 
    load_weno_stencil(buffer, shift, dir, func::Bool = false)

Stencils for WENO reconstruction calculations

The first argument is the `buffer`, not the `order`! 
- `order = 2 * buffer - 1` for WENO reconstruction
   
Examples
========

```jldoctest
julia> using Oceananigans.Advection: load_weno_stencil

julia> load_weno_stencil(3, :x)
:((ψ[i + -3, j, k], ψ[i + -2, j, k], ψ[i + -1, j, k], ψ[i + 0, j, k], ψ[i + 1, j, k], ψ[i + 2, j, k]))

julia> load_weno_stencil(2, :x)
:((ψ[i + -2, j, k], ψ[i + -1, j, k], ψ[i + 0, j, k], ψ[i + 1, j, k]))

"""
@inline function load_weno_stencil(buffer, dir, func::Bool = false) 
    N = buffer * 2 - 1
    stencil = Vector(undef, N+1)

    for (idx, c) in enumerate(-buffer:buffer-1)
        if func 
            stencil[idx] =  dir == :x ? 
                            :(ψ(i + $c, j, k, args...)) :
                            dir == :y ?
                            :(ψ(i, j + $c, k, args...)) :
                            :(ψ(i, j, k + $c, args...))
        else    
            stencil[idx] =  dir == :x ? 
                            :(ψ[i + $c, j, k]) :
                            dir == :y ?
                            :(ψ[i, j + $c, k]) :
                            :(ψ[i, j, k + $c])
        end             
    end

    return :($(stencil...),)
end

# Stencils for left and right biased reconstruction ((ψ̅ᵢ₋ᵣ₊ⱼ for j in 0:k) for r in 0:k) to calculate v̂ᵣ = ∑ⱼ(cᵣⱼψ̅ᵢ₋ᵣ₊ⱼ) 
# where `k = N - 1`. Coefficients (cᵣⱼ for j in 0:N) for stencil r are given by `coeff_side_p(scheme, Val(r), ...)`
for dir in (:x, :y, :z), (T, f) in zip((:Any, :Function), (false, true))
    stencil = Symbol(:weno_stencil_, dir)
    @eval begin
        @inline function $stencil(i, j, k, ::WENO{2}, bias, ψ::$T, args...) 
            S = @inbounds $(load_weno_stencil(2, dir, f))
            return S₀₂(S, bias), S₁₂(S, bias)
        end

        @inline function $stencil(i, j, k, ::WENO{3}, bias, ψ::$T, args...) 
            S = @inbounds $(load_weno_stencil(3, dir, f))
            return S₀₃(S, bias), S₁₃(S, bias), S₂₃(S, bias)
        end

        @inline function $stencil(i, j, k, ::WENO{4}, bias, ψ::$T, args...) 
            S = @inbounds $(load_weno_stencil(4, dir, f))
            return S₀₄(S, bias), S₁₄(S, bias), S₂₄(S, bias), S₃₄(S, bias)
        end

        @inline function $stencil(i, j, k, ::WENO{5}, bias, ψ::$T, args...) 
            S = @inbounds $(load_weno_stencil(5, dir, f))
            return S₀₅(S, bias), S₁₅(S, bias), S₂₅(S, bias), S₃₅(S, bias), S₄₅(S, bias)
        end

        @inline function $stencil(i, j, k, ::WENO{6}, bias, ψ::$T, args...) 
            S = @inbounds $(load_weno_stencil(6, dir, f))
            return S₀₆(S, bias), S₁₆(S, bias), S₂₆(S, bias), S₃₆(S, bias), S₄₆(S, bias), S₅₆(S, bias)
        end
    end
end

# WENO stencils
@inline S₀₂(S, bias) = @inbounds ifelse(bias == LeftBias(), (S[2], S[3]), (S[3], S[2]))
@inline S₁₂(S, bias) = @inbounds ifelse(bias == LeftBias(), (S[1], S[2]), (S[4], S[3]))

@inline S₀₃(S, bias) = @inbounds ifelse(bias == LeftBias(), (S[3], S[4], S[5]), (S[4], S[3], S[2]))
@inline S₁₃(S, bias) = @inbounds ifelse(bias == LeftBias(), (S[2], S[3], S[4]), (S[5], S[4], S[3]))
@inline S₂₃(S, bias) = @inbounds ifelse(bias == LeftBias(), (S[1], S[2], S[3]), (S[6], S[5], S[4]))

@inline S₀₄(S, bias) = @inbounds ifelse(bias == LeftBias(), (S[4], S[5], S[6], S[7]), (S[5], S[4], S[3], S[2]))
@inline S₁₄(S, bias) = @inbounds ifelse(bias == LeftBias(), (S[3], S[4], S[5], S[6]), (S[6], S[5], S[4], S[3]))
@inline S₂₄(S, bias) = @inbounds ifelse(bias == LeftBias(), (S[2], S[3], S[4], S[5]), (S[7], S[6], S[5], S[4]))
@inline S₃₄(S, bias) = @inbounds ifelse(bias == LeftBias(), (S[1], S[2], S[3], S[4]), (S[8], S[7], S[6], S[5]))

@inline S₀₅(S, bias) = @inbounds ifelse(bias == LeftBias(), (S[5], S[6], S[7], S[8], S[9]), (S[6],  S[5], S[4], S[3], S[2]))
@inline S₁₅(S, bias) = @inbounds ifelse(bias == LeftBias(), (S[4], S[5], S[6], S[7], S[8]), (S[7],  S[6], S[5], S[4], S[3]))
@inline S₂₅(S, bias) = @inbounds ifelse(bias == LeftBias(), (S[3], S[4], S[5], S[6], S[7]), (S[8],  S[7], S[6], S[5], S[4]))
@inline S₃₅(S, bias) = @inbounds ifelse(bias == LeftBias(), (S[2], S[3], S[4], S[5], S[6]), (S[9],  S[8], S[7], S[6], S[5]))
@inline S₄₅(S, bias) = @inbounds ifelse(bias == LeftBias(), (S[1], S[2], S[3], S[4], S[5]), (S[10], S[9], S[8], S[7], S[6]))

@inline S₀₆(S, bias) = @inbounds ifelse(bias == LeftBias(), (S[6], S[7], S[8], S[9], S[10], S[11]), (S[7],  S[6],  S[5],  S[4], S[3], S[2]))
@inline S₁₆(S, bias) = @inbounds ifelse(bias == LeftBias(), (S[5], S[6], S[7], S[8], S[9],  S[10]), (S[8],  S[7],  S[6],  S[5], S[4], S[3]))
@inline S₂₆(S, bias) = @inbounds ifelse(bias == LeftBias(), (S[4], S[5], S[6], S[7], S[8],  S[9]),  (S[9],  S[8],  S[7],  S[6], S[5], S[4]))
@inline S₃₆(S, bias) = @inbounds ifelse(bias == LeftBias(), (S[3], S[4], S[5], S[6], S[7],  S[8]),  (S[10], S[9],  S[8],  S[7], S[6], S[5]))
@inline S₄₆(S, bias) = @inbounds ifelse(bias == LeftBias(), (S[2], S[3], S[4], S[5], S[6],  S[7]),  (S[11], S[10], S[9],  S[8], S[7], S[6]))
@inline S₅₆(S, bias) = @inbounds ifelse(bias == LeftBias(), (S[1], S[2], S[3], S[4], S[5],  S[6]),  (S[12], S[11], S[10], S[9], S[8], S[7]))

# Stencil for vector invariant calculation of smoothness indicators in the horizontal direction
# Parallel to the interpolation direction! (same as left/right stencil)
@inline tangential_stencil_u(i, j, k, scheme, bias, ::Val{1}, u) = @inbounds weno_stencil_x(i, j, k, scheme, bias, ℑyᵃᶠᵃ, u)
@inline tangential_stencil_u(i, j, k, scheme, bias, ::Val{2}, u) = @inbounds weno_stencil_y(i, j, k, scheme, bias, ℑyᵃᶠᵃ, u)
@inline tangential_stencil_v(i, j, k, scheme, bias, ::Val{1}, v) = @inbounds weno_stencil_x(i, j, k, scheme, bias, ℑxᶠᵃᵃ, v)
@inline tangential_stencil_v(i, j, k, scheme, bias, ::Val{2}, v) = @inbounds weno_stencil_y(i, j, k, scheme, bias, ℑxᶠᵃᵃ, v)

# Trick to force compilation of Val(stencil-1) and avoid loops on the GPU
@inline function metaprogrammed_stencil_sum(buffer)
    elem = Vector(undef, buffer)
    for stencil = 1:buffer
        elem[stencil] = :(@inbounds w[$stencil] * func(scheme, bias, Val($(stencil-1)), ψ[$stencil], cT, Val(val), idx, loc))
    end

    return Expr(:call, :+, elem...)
end

# Calculation of WENO reconstructed value v⋆ = ∑ᵣ(wᵣv̂ᵣ)
for buffer in [2, 3, 4, 5, 6]
    @eval begin
        @inline stencil_sum(scheme::WENO{$buffer}, bias, ψ, w, func, cT, val, idx, loc) = @inbounds $(metaprogrammed_stencil_sum(buffer))
    end
end

# Interpolation functions
for (interp, dir, val, cT) in zip([:xᶠᵃᵃ, :yᵃᶠᵃ, :zᵃᵃᶠ], [:x, :y, :z], [1, 2, 3], [:XT, :YT, :ZT]) 
    interpolate_func = Symbol(:inner_biased_interpolate_, interp)
    stencil          = Symbol(:weno_stencil_, dir)
    
    @eval begin
        @inline function $interpolate_func(i, j, k, grid, 
                                            scheme::WENO{N, FT, XT, YT, ZT}, bias,
                                            ψ, idx, loc, args...) where {N, FT, XT, YT, ZT}
            @inbounds begin
                ψₜ = $stencil(i, j, k, scheme, bias, ψ, grid, args...)
                w = biased_weno_weights(ψₜ, scheme, bias, Val($val), Nothing, args...)
                return stencil_sum(scheme, bias, ψₜ, w, biased_p, $cT, $val, idx, loc)
            end
        end

        @inline function $interpolate_func(i, j, k, grid, 
                                            scheme::WENO{N, FT, XT, YT, ZT}, bias, 
                                            ψ, idx, loc, VI::AbstractSmoothnessStencil, args...) where {N, FT, XT, YT, ZT}

            @inbounds begin
                ψₜ = $stencil(i, j, k, scheme, bias, ψ, grid, args...)
                w = biased_weno_weights(ψₜ, scheme, bias, Val($val), VI, args...)
                return stencil_sum(scheme, bias, ψₜ, w, biased_p, $cT, $val, idx, loc)
            end
        end

        @inline function $interpolate_func(i, j, k, grid, 
                                            scheme::WENO{N, FT, XT, YT, ZT}, bias, 
                                            ψ, idx, loc, VI::VelocityStencil, u, v, args...) where {N, FT, XT, YT, ZT}

            @inbounds begin
                ψₜ = $stencil(i, j, k, scheme, bias, ψ, grid, u, v, args...)
                w = biased_weno_weights((i, j, k), scheme, bias, Val($val), VI, u, v)
                return stencil_sum(scheme, bias, ψₜ, w, biased_p, $cT, $val, idx, loc)
            end
        end

        @inline function $interpolate_func(i, j, k, grid, 
                                            scheme::WENO{N, FT, XT, YT, ZT}, bias, 
                                            ψ, idx, loc, VI::FunctionStencil, args...) where {N, FT, XT, YT, ZT}

            @inbounds begin
                ψₜ = $stencil(i, j, k, scheme, bias, ψ,       grid, args...)
                ψₛ = $stencil(i, j, k, scheme, bias, VI.func, grid, args...)
                w = biased_weno_weights(ψₛ, scheme, bias, Val($val), VI, args...)
                return stencil_sum(scheme, bias, ψₜ, w, biased_p, $cT, $val, idx, loc)
            end
        end
    end
end
