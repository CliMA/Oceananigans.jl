using Oceananigans.Fields: Field
using Oceananigans.Operators: ℑyᵃᶠᵃ, ℑxᶠᵃᵃ
using KernelAbstractions: @localmem, @index, @uniform, @groupsize

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
_left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, reconstruced_function::F, smoothness_stencil, args...) where F<:Function
```

For scalar reconstructions 
```julia
_left_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, reconstruced_field::F) where F<:AbstractField
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

const ε = 1e-8

# Optimal values taken from
# Balsara & Shu, "Monotonicity Preserving Weighted Essentially Non-oscillatory Schemes with Inceasingly High Order of Accuracy"
@inline coeff_left(::WENO{2}, ::Val{0}) = 2/3
@inline coeff_left(::WENO{2}, ::Val{1}) = 1/3

@inline coeff_left(::WENO{3}, ::Val{0}) = 3/10
@inline coeff_left(::WENO{3}, ::Val{1}) = 3/5
@inline coeff_left(::WENO{3}, ::Val{2}) = 1/10

@inline coeff_left(::WENO{4}, ::Val{0}) = 4/35
@inline coeff_left(::WENO{4}, ::Val{1}) = 18/35
@inline coeff_left(::WENO{4}, ::Val{2}) = 12/35
@inline coeff_left(::WENO{4}, ::Val{3}) = 1/35

@inline coeff_left(::WENO{5}, ::Val{0}) = 5/126
@inline coeff_left(::WENO{5}, ::Val{1}) = 20/63
@inline coeff_left(::WENO{5}, ::Val{2}) = 10/21
@inline coeff_left(::WENO{5}, ::Val{3}) = 10/63
@inline coeff_left(::WENO{5}, ::Val{4}) = 1/126

@inline coeff_left(::WENO{6}, ::Val{0}) = 1/77
@inline coeff_left(::WENO{6}, ::Val{1}) = 25/154
@inline coeff_left(::WENO{6}, ::Val{2}) = 100/231
@inline coeff_left(::WENO{6}, ::Val{3}) = 25/77
@inline coeff_left(::WENO{6}, ::Val{4}) = 5/77
@inline coeff_left(::WENO{6}, ::Val{5}) = 1/462

for buffer in [2, 3, 4, 5, 6]
    for stencil in [0, 1, 2, 3, 4, 5]
        @eval @inline coeff_right(scheme::WENO{$buffer}, ::Val{$stencil}) = @inbounds coeff_left(scheme, Val($(buffer-stencil-1)))
    end
end

# _UNIFORM_ smoothness coefficients (stretched smoothness coefficients are to be fixed!)
@inline smoothness_coefficients(::WENO{2, FT}, ::Val{0}) where FT = @inbounds FT.((1, -2, 1))
@inline smoothness_coefficients(::WENO{2, FT}, ::Val{1}) where FT = @inbounds FT.((1, -2, 1))

@inline smoothness_coefficients(::WENO{3, FT}, ::Val{0}) where FT = @inbounds FT.((10, -31, 11, 25, -19,  4))
@inline smoothness_coefficients(::WENO{3, FT}, ::Val{1}) where FT = @inbounds FT.((4,  -13, 5,  13, -13,  4))
@inline smoothness_coefficients(::WENO{3, FT}, ::Val{2}) where FT = @inbounds FT.((4,  -19, 11, 25, -31, 10))

@inline smoothness_coefficients(::WENO{4, FT}, ::Val{0}) where FT = @inbounds FT.((2.107,  -9.402, 7.042, -1.854, 11.003,  -17.246,  4.642,  7.043,  -3.882, 0.547))
@inline smoothness_coefficients(::WENO{4, FT}, ::Val{1}) where FT = @inbounds FT.((0.547,  -2.522, 1.922, -0.494,  3.443,  - 5.966,  1.602,  2.843,  -1.642, 0.267))
@inline smoothness_coefficients(::WENO{4, FT}, ::Val{2}) where FT = @inbounds FT.((0.267,  -1.642, 1.602, -0.494,  2.843,  - 5.966,  1.922,  3.443,  -2.522, 0.547))
@inline smoothness_coefficients(::WENO{4, FT}, ::Val{3}) where FT = @inbounds FT.((0.547,  -3.882, 4.642, -1.854,  7.043,  -17.246,  7.042, 11.003,  -9.402, 2.107))

@inline smoothness_coefficients(::WENO{5, FT}, ::Val{0}) where FT = @inbounds FT.((1.07918,  -6.49501, 7.58823, -4.11487,  0.86329,  10.20563, -24.62076, 13.58458, -2.88007, 15.21393, -17.04396, 3.64863,  4.82963, -2.08501, 0.22658)) 
@inline smoothness_coefficients(::WENO{5, FT}, ::Val{1}) where FT = @inbounds FT.((0.22658,  -1.40251, 1.65153, -0.88297,  0.18079,   2.42723,  -6.11976,  3.37018, -0.70237,  4.06293,  -4.64976, 0.99213,  1.38563, -0.60871, 0.06908)) 
@inline smoothness_coefficients(::WENO{5, FT}, ::Val{2}) where FT = @inbounds FT.((0.06908,  -0.51001, 0.67923, -0.38947,  0.08209,   1.04963,  -2.99076,  1.79098, -0.38947,  2.31153,  -2.99076, 0.67923,  1.04963, -0.51001, 0.06908)) 
@inline smoothness_coefficients(::WENO{5, FT}, ::Val{3}) where FT = @inbounds FT.((0.06908,  -0.60871, 0.99213, -0.70237,  0.18079,   1.38563,  -4.64976,  3.37018, -0.88297,  4.06293,  -6.11976, 1.65153,  2.42723, -1.40251, 0.22658)) 
@inline smoothness_coefficients(::WENO{5, FT}, ::Val{4}) where FT = @inbounds FT.((0.22658,  -2.08501, 3.64863, -2.88007,  0.86329,   4.82963, -17.04396, 13.58458, -4.11487, 15.21393, -24.62076, 7.58823, 10.20563, -6.49501, 1.07918)) 

@inline smoothness_coefficients(::WENO{6, FT}, ::Val{0}) where FT = @inbounds FT.((0.6150211, -4.7460464, 7.6206736, -6.3394124, 2.7060170, -0.4712740,  9.4851237, -31.1771244, 26.2901672, -11.3206788,  1.9834350, 26.0445372, -44.4003904, 19.2596472, -3.3918804, 19.0757572, -16.6461044, 2.9442256, 3.6480687, -1.2950184, 0.1152561)) 
@inline smoothness_coefficients(::WENO{6, FT}, ::Val{1}) where FT = @inbounds FT.((0.1152561, -0.9117992, 1.4742480, -1.2183636, 0.5134574, -0.0880548,  1.9365967,  -6.5224244,  5.5053752,  -2.3510468,  0.4067018,  5.6662212,  -9.7838784,  4.2405032, -0.7408908,  4.3093692,  -3.7913324, 0.6694608, 0.8449957, -0.3015728, 0.0271779)) 
@inline smoothness_coefficients(::WENO{6, FT}, ::Val{2}) where FT = @inbounds FT.((0.0271779, -0.2380800, 0.4086352, -0.3462252, 0.1458762, -0.0245620,  0.5653317,  -2.0427884,  1.7905032,  -0.7727988,  0.1325006,  1.9510972,  -3.5817664,  1.5929912, -0.2792660,  1.7195652,  -1.5880404, 0.2863984, 0.3824847, -0.1429976, 0.0139633)) 
@inline smoothness_coefficients(::WENO{6, FT}, ::Val{3}) where FT = @inbounds FT.((0.0139633, -0.1429976, 0.2863984, -0.2792660, 0.1325006, -0.0245620,  0.3824847,  -1.5880404,  1.5929912,  -0.7727988,  0.1458762,  1.7195652,  -3.5817664,  1.7905032, -0.3462252,  1.9510972,  -2.0427884, 0.4086352, 0.5653317, -0.2380800, 0.0271779)) 
@inline smoothness_coefficients(::WENO{6, FT}, ::Val{4}) where FT = @inbounds FT.((0.0271779, -0.3015728, 0.6694608, -0.7408908, 0.4067018, -0.0880548,  0.8449957,  -3.7913324,  4.2405032,  -2.3510468,  0.5134574,  4.3093692,  -9.7838784,  5.5053752, -1.2183636,  5.6662212,  -6.5224244, 1.4742480, 1.9365967, -0.9117992, 0.1152561)) 
@inline smoothness_coefficients(::WENO{6, FT}, ::Val{5}) where FT = @inbounds FT.((0.1152561, -1.2950184, 2.9442256, -3.3918804, 1.9834350, -0.4712740,  3.6480687, -16.6461044, 19.2596472, -11.3206788,  2.7060170, 19.0757572, -44.4003904, 26.2901672, -6.3394124, 26.0445372, -31.1771244, 7.6206736, 9.4851237, -4.7460464, 0.6150211)) 

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
        stencil_sum   = Expr(:call, :+, (:(C[$(c_idx + i - stencil)] * ψ[$i]) for i in stencil:buffer)...)
        elem[stencil] = :(ψ[$stencil] * $stencil_sum)
        c_idx += buffer - stencil + 1
    end

    elem[buffer] = :(ψ[$buffer] * ψ[$buffer] * C[$c_idx])
    
    return Expr(:call, :+, elem...)
end

# Smoothness indicators for stencil `stencil` for left and right biased reconstruction
for buffer in [2, 3, 4, 5, 6]
    @eval begin
        @inline smoothness_sum(scheme::WENO{$buffer}, ψ, C) = @inbounds @fastmath $(metaprogrammed_smoothness_sum(buffer))
    end

    for stencil in [0, 1, 2, 3, 4, 5]
        @eval begin
            @inline  left_biased_β(ψ, scheme::WENO{$buffer}, ::Val{$stencil}) = smoothness_sum(scheme, ψ, smoothness_coefficients(scheme, Val($stencil)))
            @inline right_biased_β(ψ, scheme::WENO{$buffer}, ::Val{$stencil}) = smoothness_sum(scheme, ψ, smoothness_coefficients(scheme, Val($stencil)))
        end

        # ENO coefficients for uniform direction (when T<:Nothing) and stretched directions (when T<:Any) 
        @eval begin
            # uniform coefficients are independent on direction and location
            @inline  coeff_left_p(scheme::WENO{$buffer, FT}, ::Val{$stencil}, ::Type{Nothing}, args...) where FT = @inbounds FT.($(stencil_coefficients(50, stencil  , collect(1:100), collect(1:100); order = buffer)))
            @inline coeff_right_p(scheme::WENO{$buffer, FT}, ::Val{$stencil}, ::Type{Nothing}, args...) where FT = @inbounds FT.($(stencil_coefficients(50, stencil-1, collect(1:100), collect(1:100); order = buffer)))

            # stretched coefficients are retrieved from precalculated coefficients
            @inline  coeff_left_p(scheme::WENO{$buffer}, ::Val{$stencil}, T, dir, i, loc) = @inbounds retrieve_coeff(scheme, $stencil,     dir, i, loc)
            @inline coeff_right_p(scheme::WENO{$buffer}, ::Val{$stencil}, T, dir, i, loc) = @inbounds retrieve_coeff(scheme, $(stencil-1), dir, i, loc)
        end
    
        # left biased and right biased reconstruction value for each stencil
        @eval begin
            @inline  left_biased_p(scheme::WENO{$buffer}, ::Val{$stencil}, ψ, T, dir, i, loc) = @inbounds  sum(coeff_left_p(scheme, Val($stencil), T, dir, i, loc) .* ψ)
            @inline right_biased_p(scheme::WENO{$buffer}, ::Val{$stencil}, ψ, T, dir, i, loc) = @inbounds sum(coeff_right_p(scheme, Val($stencil), T, dir, i, loc) .* ψ)
        end
    end
end

# Global smoothness indicator τ₂ᵣ₋₁ taken from "Accuracy of the weighted essentially non-oscillatory conservative finite difference schemes", Don & Borges, 2013
@inline global_smoothness_indicator(::Val{2}, β) = @inbounds abs(β[1] - β[2])
@inline global_smoothness_indicator(::Val{3}, β) = @inbounds abs(β[1] - β[3])
@inline global_smoothness_indicator(::Val{4}, β) = @inbounds abs(β[1] +  3β[2] - 3β[3] -  β[4])
@inline global_smoothness_indicator(::Val{5}, β) = @inbounds abs(β[1] +  2β[2] - 6β[3] + 2β[4] + β[5])
@inline global_smoothness_indicator(::Val{6}, β) = @inbounds abs(β[1] +   β[2] - 8β[3] + 8β[4] - β[5] - β[6])

@inline add_global_smoothness(β, ::Val{2}, ::Val{1}) = + β
@inline add_global_smoothness(β, ::Val{2}, ::Val{2}) = - β

@inline add_global_smoothness(β, ::Val{3}, ::Val{1}) = + β
@inline add_global_smoothness(β, ::Val{3}, ::Val{2}) = 0
@inline add_global_smoothness(β, ::Val{3}, ::Val{3}) = - β

@inline add_global_smoothness(β, ::Val{4}, ::Val{1}) = +  β
@inline add_global_smoothness(β, ::Val{4}, ::Val{2}) = + 3β
@inline add_global_smoothness(β, ::Val{4}, ::Val{3}) = - 3β
@inline add_global_smoothness(β, ::Val{4}, ::Val{4}) = -  β

@inline add_global_smoothness(β, ::Val{5}, ::Val{1}) = +  β
@inline add_global_smoothness(β, ::Val{5}, ::Val{2}) = + 2β
@inline add_global_smoothness(β, ::Val{5}, ::Val{3}) = - 6β
@inline add_global_smoothness(β, ::Val{5}, ::Val{4}) = + 2β
@inline add_global_smoothness(β, ::Val{5}, ::Val{5}) = +  β

@inline add_global_smoothness(β, ::Val{6}, ::Val{1}) = +  β
@inline add_global_smoothness(β, ::Val{6}, ::Val{2}) = +  β
@inline add_global_smoothness(β, ::Val{6}, ::Val{3}) = - 8β
@inline add_global_smoothness(β, ::Val{6}, ::Val{4}) = + 8β
@inline add_global_smoothness(β, ::Val{6}, ::Val{5}) = -  β
@inline add_global_smoothness(β, ::Val{6}, ::Val{6}) = -  β

""" 
    calc_weno_stencil(buffer, shift, dir, func::Bool = false)

Stencils for WENO reconstruction calculations

The first argument is the `buffer`, not the `order`! 
- `order = 2 * buffer - 1` for WENO reconstruction
   
Examples
========

```jldoctest
julia> using Oceananigans.Advection: calc_weno_stencil

julia> calc_weno_stencil(3, :left, :x)
:(((ψ[i + -1, j, k], ψ[i + 0, j, k], ψ[i + 1, j, k]), (ψ[i + -2, j, k], ψ[i + -1, j, k], ψ[i + 0, j, k]), (ψ[i + -3, j, k], ψ[i + -2, j, k], ψ[i + -1, j, k])))

julia> calc_weno_stencil(2, :right, :x)
:(((ψ[i + 0, j, k], ψ[i + 1, j, k]), (ψ[i + -1, j, k], ψ[i + 0, j, k])))

"""
@inline function calc_weno_stencil(buffer, shift, dir, func::Bool = false) 
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
    return stencil_full
end

# Stencils for left and right biased reconstruction ((ψ̅ᵢ₋ᵣ₊ⱼ for j in 0:k) for r in 0:k) to calculate v̂ᵣ = ∑ⱼ(cᵣⱼψ̅ᵢ₋ᵣ₊ⱼ) 
# where `k = N - 1`. Coefficients (cᵣⱼ for j in 0:N) for stencil r are given by `coeff_side_p(scheme, Val(r), ...)`
for side in (:left, :right), dir in (:xᶠᵃᵃ, :yᵃᶠᵃ, :zᵃᵃᶠ)
    retrieve_stencil = Symbol(side, :_stencil_, dir)
    for buffer in [2, 3, 4, 5, 6]
        for stencil in 1:buffer
            @eval begin
                @inline $retrieve_stencil(i, j, k, ::WENO{$buffer}, ::Val{$stencil}, ψ, args...)           = @inbounds $(calc_weno_stencil(buffer, side, dir, false)[stencil])
                @inline $retrieve_stencil(i, j, k, ::WENO{$buffer}, ::Val{$stencil}, ψ::Function, args...) = @inbounds $(calc_weno_stencil(buffer, side, dir,  true)[stencil])
            end
        end
    end
end

# Stencil for vector invariant calculation of smoothness indicators in the horizontal direction
# Parallel to the interpolation direction! (same as left/right stencil)
@inline tangential_left_stencil_u(i, j, k, scheme, stencil, ::Val{1}, grid, u) = @inbounds left_stencil_xᶠᵃᵃ(i, j, k, scheme, stencil, ℑyᵃᶠᵃ, grid, u)
@inline tangential_left_stencil_u(i, j, k, scheme, stencil, ::Val{2}, grid, u) = @inbounds left_stencil_yᵃᶠᵃ(i, j, k, scheme, stencil, ℑyᵃᶠᵃ, grid, u)
@inline tangential_left_stencil_v(i, j, k, scheme, stencil, ::Val{1}, grid, v) = @inbounds left_stencil_xᶠᵃᵃ(i, j, k, scheme, stencil, ℑxᶠᵃᵃ, grid, v)
@inline tangential_left_stencil_v(i, j, k, scheme, stencil, ::Val{2}, grid, v) = @inbounds left_stencil_yᵃᶠᵃ(i, j, k, scheme, stencil, ℑxᶠᵃᵃ, grid, v)

@inline tangential_right_stencil_u(i, j, k, scheme, stencil, ::Val{1}, grid, u) = @inbounds right_stencil_xᶠᵃᵃ(i, j, k, scheme, stencil, ℑyᵃᶠᵃ, grid, u)
@inline tangential_right_stencil_u(i, j, k, scheme, stencil, ::Val{2}, grid, u) = @inbounds right_stencil_yᵃᶠᵃ(i, j, k, scheme, stencil, ℑyᵃᶠᵃ, grid, u)
@inline tangential_right_stencil_v(i, j, k, scheme, stencil, ::Val{1}, grid, v) = @inbounds right_stencil_xᶠᵃᵃ(i, j, k, scheme, stencil, ℑxᶠᵃᵃ, grid, v)
@inline tangential_right_stencil_v(i, j, k, scheme, stencil, ::Val{2}, grid, v) = @inbounds right_stencil_yᵃᶠᵃ(i, j, k, scheme, stencil, ℑxᶠᵃᵃ, grid, v)

@inline new_stencil_left_xᶠᵃᵃ(i, j, k, ::WENO{N}, ::Val{2}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i-2, j, k, grid, args...), ψs[1:N-1]...)
@inline new_stencil_left_xᶠᵃᵃ(i, j, k, ::WENO{N}, ::Val{3}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i-3, j, k, grid, args...), ψs[1:N-1]...)
@inline new_stencil_left_xᶠᵃᵃ(i, j, k, ::WENO{N}, ::Val{4}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i-4, j, k, grid, args...), ψs[1:N-1]...)
@inline new_stencil_left_xᶠᵃᵃ(i, j, k, ::WENO{N}, ::Val{5}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i-5, j, k, grid, args...), ψs[1:N-1]...)
@inline new_stencil_left_xᶠᵃᵃ(i, j, k, ::WENO{N}, ::Val{6}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i-6, j, k, grid, args...), ψs[1:N-1]...)

@inline new_stencil_left_yᵃᶠᵃ(i, j, k, ::WENO{N}, ::Val{2}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i, j-2, k, grid, args...), ψs[1:N-1]...)
@inline new_stencil_left_yᵃᶠᵃ(i, j, k, ::WENO{N}, ::Val{3}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i, j-3, k, grid, args...), ψs[1:N-1]...)
@inline new_stencil_left_yᵃᶠᵃ(i, j, k, ::WENO{N}, ::Val{4}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i, j-4, k, grid, args...), ψs[1:N-1]...)
@inline new_stencil_left_yᵃᶠᵃ(i, j, k, ::WENO{N}, ::Val{5}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i, j-5, k, grid, args...), ψs[1:N-1]...)
@inline new_stencil_left_yᵃᶠᵃ(i, j, k, ::WENO{N}, ::Val{6}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i, j-6, k, grid, args...), ψs[1:N-1]...)

@inline new_stencil_left_zᵃᵃᶠ(i, j, k, ::WENO{N}, ::Val{2}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i, j, k-2, grid, args...), ψs[1:N-1]...)
@inline new_stencil_left_zᵃᵃᶠ(i, j, k, ::WENO{N}, ::Val{3}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i, j, k-3, grid, args...), ψs[1:N-1]...)
@inline new_stencil_left_zᵃᵃᶠ(i, j, k, ::WENO{N}, ::Val{4}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i, j, k-4, grid, args...), ψs[1:N-1]...)
@inline new_stencil_left_zᵃᵃᶠ(i, j, k, ::WENO{N}, ::Val{5}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i, j, k-5, grid, args...), ψs[1:N-1]...)
@inline new_stencil_left_zᵃᵃᶠ(i, j, k, ::WENO{N}, ::Val{6}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i, j, k-6, grid, args...), ψs[1:N-1]...)

@inline new_stencil_left_xᶠᵃᵃ(i, j, k, ::WENO{N}, ::Val{2}, ψs, ψ, args...) where N = @inbounds (ψ[i-2, j, k], ψs[1:N-1]...)
@inline new_stencil_left_xᶠᵃᵃ(i, j, k, ::WENO{N}, ::Val{3}, ψs, ψ, args...) where N = @inbounds (ψ[i-3, j, k], ψs[1:N-1]...)
@inline new_stencil_left_xᶠᵃᵃ(i, j, k, ::WENO{N}, ::Val{4}, ψs, ψ, args...) where N = @inbounds (ψ[i-4, j, k], ψs[1:N-1]...)
@inline new_stencil_left_xᶠᵃᵃ(i, j, k, ::WENO{N}, ::Val{5}, ψs, ψ, args...) where N = @inbounds (ψ[i-5, j, k], ψs[1:N-1]...)
@inline new_stencil_left_xᶠᵃᵃ(i, j, k, ::WENO{N}, ::Val{6}, ψs, ψ, args...) where N = @inbounds (ψ[i-6, j, k], ψs[1:N-1]...)

@inline new_stencil_left_yᵃᶠᵃ(i, j, k, ::WENO{N}, ::Val{2}, ψs, ψ, args...) where N = @inbounds (ψ[i, j-2, k], ψs[1:N-1]...)
@inline new_stencil_left_yᵃᶠᵃ(i, j, k, ::WENO{N}, ::Val{3}, ψs, ψ, args...) where N = @inbounds (ψ[i, j-3, k], ψs[1:N-1]...)
@inline new_stencil_left_yᵃᶠᵃ(i, j, k, ::WENO{N}, ::Val{4}, ψs, ψ, args...) where N = @inbounds (ψ[i, j-4, k], ψs[1:N-1]...)
@inline new_stencil_left_yᵃᶠᵃ(i, j, k, ::WENO{N}, ::Val{5}, ψs, ψ, args...) where N = @inbounds (ψ[i, j-5, k], ψs[1:N-1]...)
@inline new_stencil_left_yᵃᶠᵃ(i, j, k, ::WENO{N}, ::Val{6}, ψs, ψ, args...) where N = @inbounds (ψ[i, j-6, k], ψs[1:N-1]...)

@inline new_stencil_left_zᵃᵃᶠ(i, j, k, ::WENO{N}, ::Val{2}, ψs, ψ, args...) where N = @inbounds (ψ[i, j, k-2], ψs[1:N-1]...)
@inline new_stencil_left_zᵃᵃᶠ(i, j, k, ::WENO{N}, ::Val{3}, ψs, ψ, args...) where N = @inbounds (ψ[i, j, k-3], ψs[1:N-1]...)
@inline new_stencil_left_zᵃᵃᶠ(i, j, k, ::WENO{N}, ::Val{4}, ψs, ψ, args...) where N = @inbounds (ψ[i, j, k-4], ψs[1:N-1]...)
@inline new_stencil_left_zᵃᵃᶠ(i, j, k, ::WENO{N}, ::Val{5}, ψs, ψ, args...) where N = @inbounds (ψ[i, j, k-5], ψs[1:N-1]...)
@inline new_stencil_left_zᵃᵃᶠ(i, j, k, ::WENO{N}, ::Val{6}, ψs, ψ, args...) where N = @inbounds (ψ[i, j, k-6], ψs[1:N-1]...)

@inline new_stencil_right_xᶠᵃᵃ(i, j, k, ::WENO{N}, ::Val{2}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i-1, j, k, grid, args...), ψs[1:N-1]...)
@inline new_stencil_right_xᶠᵃᵃ(i, j, k, ::WENO{N}, ::Val{3}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i-2, j, k, grid, args...), ψs[1:N-1]...)
@inline new_stencil_right_xᶠᵃᵃ(i, j, k, ::WENO{N}, ::Val{4}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i-3, j, k, grid, args...), ψs[1:N-1]...)
@inline new_stencil_right_xᶠᵃᵃ(i, j, k, ::WENO{N}, ::Val{5}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i-4, j, k, grid, args...), ψs[1:N-1]...)
@inline new_stencil_right_xᶠᵃᵃ(i, j, k, ::WENO{N}, ::Val{6}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i-5, j, k, grid, args...), ψs[1:N-1]...)

@inline new_stencil_right_yᵃᶠᵃ(i, j, k, ::WENO{N}, ::Val{2}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i, j-1, k, grid, args...), ψs[1:N-1]...)
@inline new_stencil_right_yᵃᶠᵃ(i, j, k, ::WENO{N}, ::Val{3}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i, j-2, k, grid, args...), ψs[1:N-1]...)
@inline new_stencil_right_yᵃᶠᵃ(i, j, k, ::WENO{N}, ::Val{4}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i, j-3, k, grid, args...), ψs[1:N-1]...)
@inline new_stencil_right_yᵃᶠᵃ(i, j, k, ::WENO{N}, ::Val{5}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i, j-4, k, grid, args...), ψs[1:N-1]...)
@inline new_stencil_right_yᵃᶠᵃ(i, j, k, ::WENO{N}, ::Val{6}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i, j-5, k, grid, args...), ψs[1:N-1]...)

@inline new_stencil_right_zᵃᵃᶠ(i, j, k, ::WENO{N}, ::Val{2}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i, j, k-1, grid, args...), ψs[1:N-1]...)
@inline new_stencil_right_zᵃᵃᶠ(i, j, k, ::WENO{N}, ::Val{3}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i, j, k-2, grid, args...), ψs[1:N-1]...)
@inline new_stencil_right_zᵃᵃᶠ(i, j, k, ::WENO{N}, ::Val{4}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i, j, k-3, grid, args...), ψs[1:N-1]...)
@inline new_stencil_right_zᵃᵃᶠ(i, j, k, ::WENO{N}, ::Val{5}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i, j, k-4, grid, args...), ψs[1:N-1]...)
@inline new_stencil_right_zᵃᵃᶠ(i, j, k, ::WENO{N}, ::Val{6}, ψs, ψ::Function, grid, args...) where N = @inbounds (ψ(i, j, k-5, grid, args...), ψs[1:N-1]...)

@inline new_stencil_right_xᶠᵃᵃ(i, j, k, ::WENO{N}, ::Val{2}, ψs, ψ, args...) where N = @inbounds (ψ[i-1, j, k], ψs[1:N-1]...)
@inline new_stencil_right_xᶠᵃᵃ(i, j, k, ::WENO{N}, ::Val{3}, ψs, ψ, args...) where N = @inbounds (ψ[i-2, j, k], ψs[1:N-1]...)
@inline new_stencil_right_xᶠᵃᵃ(i, j, k, ::WENO{N}, ::Val{4}, ψs, ψ, args...) where N = @inbounds (ψ[i-3, j, k], ψs[1:N-1]...)
@inline new_stencil_right_xᶠᵃᵃ(i, j, k, ::WENO{N}, ::Val{5}, ψs, ψ, args...) where N = @inbounds (ψ[i-4, j, k], ψs[1:N-1]...)
@inline new_stencil_right_xᶠᵃᵃ(i, j, k, ::WENO{N}, ::Val{6}, ψs, ψ, args...) where N = @inbounds (ψ[i-5, j, k], ψs[1:N-1]...)

@inline new_stencil_right_yᵃᶠᵃ(i, j, k, ::WENO{N}, ::Val{2}, ψs, ψ, args...) where N = @inbounds (ψ[i, j-1, k], ψs[1:N-1]...)
@inline new_stencil_right_yᵃᶠᵃ(i, j, k, ::WENO{N}, ::Val{3}, ψs, ψ, args...) where N = @inbounds (ψ[i, j-2, k], ψs[1:N-1]...)
@inline new_stencil_right_yᵃᶠᵃ(i, j, k, ::WENO{N}, ::Val{4}, ψs, ψ, args...) where N = @inbounds (ψ[i, j-3, k], ψs[1:N-1]...)
@inline new_stencil_right_yᵃᶠᵃ(i, j, k, ::WENO{N}, ::Val{5}, ψs, ψ, args...) where N = @inbounds (ψ[i, j-4, k], ψs[1:N-1]...)
@inline new_stencil_right_yᵃᶠᵃ(i, j, k, ::WENO{N}, ::Val{6}, ψs, ψ, args...) where N = @inbounds (ψ[i, j-5, k], ψs[1:N-1]...)

@inline new_stencil_right_zᵃᵃᶠ(i, j, k, ::WENO{N}, ::Val{2}, ψs, ψ, args...) where N = @inbounds (ψ[i, j, k-1], ψs[1:N-1]...)
@inline new_stencil_right_zᵃᵃᶠ(i, j, k, ::WENO{N}, ::Val{3}, ψs, ψ, args...) where N = @inbounds (ψ[i, j, k-2], ψs[1:N-1]...)
@inline new_stencil_right_zᵃᵃᶠ(i, j, k, ::WENO{N}, ::Val{4}, ψs, ψ, args...) where N = @inbounds (ψ[i, j, k-3], ψs[1:N-1]...)
@inline new_stencil_right_zᵃᵃᶠ(i, j, k, ::WENO{N}, ::Val{5}, ψs, ψ, args...) where N = @inbounds (ψ[i, j, k-4], ψs[1:N-1]...)
@inline new_stencil_right_zᵃᵃᶠ(i, j, k, ::WENO{N}, ::Val{6}, ψs, ψ, args...) where N = @inbounds (ψ[i, j, k-5], ψs[1:N-1]...)

for side in [:left, :right], (dir, val, CT) in zip([:xᶠᵃᵃ, :yᵃᶠᵃ, :zᵃᵃᶠ], [1, 2, 3], [:XT, :YT, :ZT])
    weno_interpolant = Symbol(side, :_weno_interpolant_, dir)
    biased_β         = Symbol(side, :_biased_β)
    biased_p         = Symbol(side, :_biased_p)
    coeff            = Symbol(:coeff_, side) 
    stencil          = Symbol(side, :_stencil_, dir)
    stencil_u        = Symbol(:tangential_, side, :_stencil_u)
    stencil_v        = Symbol(:tangential_, side, :_stencil_v)

    @eval begin
        # Standard weno reconstruction at `i, j, k` for stencil `s` where `1 ≤ s ≤ N`
        @inline function $weno_interpolant(i::Int, j::Int, k::Int, s::Int, grid, scheme::WENO{N, FT, XT, YT, ZT}, val, ψ, idx, loc, args...) where {N, FT, XT, YT, ZT}
            
            # Retrieve stencil `s`
            ψs = $stencil(i, j, k, scheme, Val(s), ψ, grid, args...)

            # Calculate smoothness of stencil `s`
            β = $biased_β(ψs, scheme, Val(s-1))

            # Calculate the `α` coefficient of stencil `s` following a WENO-JS formulation
            C = FT($coeff(scheme, Val(s-1)))
            α = C / (β + FT(ε))^2

            # Reconstruction of `ψ` from stencil `s`
            ψ̅ = $biased_p(scheme, Val(s-1), ψs, $CT, Val(val), idx, loc) 

            return β, ψ̅, C, α
        end

        # If the smoothness stencil is not used (aka it's a `DefaultStencil`) use the same formulation as above
        @inline function $weno_interpolant(i::Int, j::Int, k::Int, s::Int, grid, scheme::WENO{N, FT, XT, YT, ZT}, val, ψ, idx, loc, ::AbstractSmoothnessStencil, args...) where {N, FT, XT, YT, ZT}
            
            # Retrieve stencil `s`
            ψs = $stencil(i, j, k, scheme, Val(s), ψ, grid, args...)
            
            # Calculate smoothness of stencil `s`
            β = $biased_β(ψs, scheme, Val(s-1))
            
            # Calculate the `α` coefficient of stencil `s` following a WENO-JS formulation
            C = FT($coeff(scheme, Val(s-1)))
            α = C / (β + FT(ε))^2
            
            # Reconstruction of `ψ` from stencil `s`
            ψ̅ = $biased_p(scheme, Val(s-1), ψs, $CT, Val(val), idx, loc) 

            return β, ψ̅, C, α
        end

        # Using velocity interpolated at `(Face, Face, Center)` to assess smoothness. 
        # Can be used only for `(Face, Face, Center)` variables like vorticity
        @inline function $weno_interpolant(i::Int, j::Int, k::Int, s::Int, grid, scheme::WENO{N, FT, XT, YT, ZT}, val, ψ, idx, loc, ::VelocityStencil, u, v, args...) where {N, FT, XT, YT, ZT}
            
            # Retrieve x-velocity stencil `s`
            ψs = $stencil_u(i, j, k, scheme, Val(s), Val(val), grid, u)
            
            # Calculate x-velocity smoothness at stencil `s`
            βu = $biased_β(ψs, scheme, Val(s-1))

            # Retrieve y-velocity stencil `s`
            ψs = $stencil_v(i, j, k, scheme, Val(s), Val(val), grid, v)
            
            # Calculate y-velocity smoothness at stencil `s`
            βv = $biased_β(ψs, scheme, Val(s-1))
            
            # total smoothness
            βᵁ = (βu + βv) / 2
            
            # Calculate the `α` coefficient of stencil `s` following a WENO-JS formulation
            C = FT($coeff(scheme, Val(s-1)))
            α = C / (βᵁ + FT(ε))^2

            # Retrieve stencil `s` and reconstruct `ψ` from stencil `s`
            ψs = $stencil(i, j, k, scheme, Val(s), ψ, grid, u, v, args...)
            ψ̅  = $biased_p(scheme, Val(s-1), ψs, $CT, Val(val), idx, loc) 

            return βᵁ, ψ̅, C, α
        end

        # The smoothness is assessed using the stencil calculated from the function `VI.func(i, j, k, grid, args...)`
        @inline function $weno_interpolant(i::Int, j::Int, k::Int, s::Int, grid, scheme::WENO{N, FT, XT, YT, ZT}, val, ψ, idx, loc, VI::FunctionStencil, args...) where {N, FT, XT, YT, ZT}
            
            # Retrieve smoothness stencil ϕ at `s`
            ψs = $stencil(i, j, k, scheme, Val(s), VI.func, grid, args...)

            # Calculate `ϕ` smoothness at `s`
            βᵠ = $biased_β(ψs, scheme, Val(s-1))

            # Calculate the `α` coefficient of stencil `s` following a WENO-JS formulation
            C  = FT($coeff(scheme, Val(s-1)))
            α  = C / (βᵠ + FT(ε))^2

            # Retrieve stencil `s` and reconstruct `ψ` from stencil `s`
            ψs = $stencil(i, j, k, scheme, Val(s), ψ, grid, args...)
            ψ̅  = $biased_p(scheme, Val(s-1), ψs, $CT, Val(val), idx, loc) 

            return βᵠ, ψ̅, C, α
        end

        #####
        ##### Picking up the stencil prior
        #####

        # Standard weno reconstruction at `i, j, k` for stencil `s` where `1 ≤ s ≤ N`
        @inline function $weno_interpolant(ψs::Tuple, s, scheme::WENO{N, FT, XT, YT, ZT}, val, idx, loc) where {N, FT, XT, YT, ZT}
            
            # Calculate smoothness of stencil `s`
            β = $biased_β(ψs, scheme, Val(s-1))

            # Calculate the `α` coefficient of stencil `s` following a WENO-JS formulation
            C = FT($coeff(scheme, Val(s-1)))
            α = C / (β + FT(ε))^2

            # Reconstruction of `ψ` from stencil `s`
            ψ̅ = $biased_p(scheme, Val(s-1), ψs, $CT, Val(val), idx, loc) 

            return β, ψ̅, C, α
        end

        # Using velocity interpolated at `(Face, Face, Center)` to assess smoothness. 
        # Can be used only for `(Face, Face, Center)` variables like vorticity
        @inline function $weno_interpolant(ψs::Tuple, us::Tuple, vs::Tuple, s, scheme::WENO{N, FT, XT, YT, ZT}, val, idx, loc) where {N, FT, XT, YT, ZT}
            
            # Calculate x-velocity smoothness at stencil `s`
            βu = $biased_β(us, scheme, Val(s-1))
            # Calculate y-velocity smoothness at stencil `s`
            βv = $biased_β(vs, scheme, Val(s-1))
            
            # total smoothness
            βᵁ = (βu + βv) / 2
            
            # Calculate the `α` coefficient of stencil `s` following a WENO-JS formulation
            C = FT($coeff(scheme, Val(s-1)))
            α = C / (βᵁ + FT(ε))^2

            # Retrieve stencil `s` and reconstruct `ψ` from stencil `s`
            ψ̅  = $biased_p(scheme, Val(s-1), ψs, $CT, Val(val), idx, loc) 

            return βᵁ, ψ̅, C, α
        end

        # The smoothness is assessed using the stencil calculated from the function `VI.func(i, j, k, grid, args...)`
        @inline function $weno_interpolant(ψs::Tuple, ϕs::Tuple, s, scheme::WENO{N, FT, XT, YT, ZT}, val, idx, loc) where {N, FT, XT, YT, ZT}
            
            # Calculate `ϕ` smoothness at `s`
            βᵠ = $biased_β(ϕs, scheme, Val(s-1))

            # Calculate the `α` coefficient of stencil `s` following a WENO-JS formulation
            C  = FT($coeff(scheme, Val(s-1)))
            α  = C / (βᵠ + FT(ε))^2

            # Retrieve stencil `s` and reconstruct `ψ` from stencil `s`
            ψ̅  = $biased_p(scheme, Val(s-1), ψs, $CT, Val(val), idx, loc) 

            return βᵠ, ψ̅, C, α
        end
    end
end
