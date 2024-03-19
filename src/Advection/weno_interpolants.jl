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

include("smoothness_coefficients.jl")

Base.show(io::IO, a::FunctionStencil) =  print(io, "FunctionStencil f = $(a.func)")

const ε = 1e-8

# ENO reconstruction procedure per stencil 
for buffer in [2, 3, 4, 5, 6]
    for stencil in collect(0:1:buffer-1)

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

# The rule for calculating smoothness indicators is the following (example WENO{4} which is seventh order) 
# ψ[1] (C[1]  * ψ[1] + C[2] * ψ[2] + C[3] * ψ[3] + C[4] * ψ[4]) + 
# ψ[2] (C[5]  * ψ[2] + C[6] * ψ[3] + C[7] * ψ[4]) + 
# ψ[3] (C[8]  * ψ[3] + C[9] * ψ[4])
# ψ[4] (C[10] * ψ[4])
# This expression is the output of metaprogrammed_smoothness_sum(4)

# Trick to force compilation of Val(stencil-1) and avoid loops on the GPU
@inline function metaprogrammed_smoothness_sum(buffer, stencil)
    elem = Vector(undef, buffer+1)
    c_idx = 1
    for s = 1:buffer - 1
        elements = Vector(undef, buffer-s+1)
        for i in s:buffer
            smoothness_coefficient = Symbol(:Cβ_, buffer, stencil, c_idx)
            coeff = eval(smoothness_coefficient)
            elements[i - s + 1] = :($coeff * ψ[$i])
            c_idx += 1
        end    
        elem[s] = :(ψ[$s] * $(Expr(:call, :+, elements...)))
    end

    smoothness_coefficient = Symbol(:Cβ_, buffer, stencil, c_idx)
    coeff = eval(smoothness_coefficient)
    elem[buffer]   = :(ψ[$buffer] * ψ[$buffer] * $coeff)
    elem[buffer+1] = ε
    return Expr(:call, :+, elem...)
end

# Shenanigans for WENO weights calculation for vector invariant formulation -> [β[i] = 0.5*(βᵤ[i] + βᵥ[i]) for i in 1:buffer]
@inline function metaprogrammed_beta_sum(buffer)
    elem = Vector(undef, buffer)
    for stencil = 1:buffer
        elem[stencil] = :((β₁[$stencil] + β₂[$stencil])/2)
    end
    return :($(elem...),)
end

# left and right biased_β calculation for scheme and stencil = 0:buffer - 1
@inline function metaprogrammed_beta_loop(buffer)
    elem = Vector(undef, buffer)
    for stencil = 1:buffer
        elem[stencil] = :(beta_value(ψ[$stencil], Val($buffer), Val($(stencil-1))))
    end

    return :($(elem...),)
end

# ZWENO α weights dᵣ * (1 + (τ₂ᵣ₋₁ / βᵣ))ᵖ)
@inline function metaprogrammed_alpha_loop(buffer, side)
    elem = Vector(undef, buffer)

    for stencil = 1:buffer
        coeff = eval(Symbol(:coeff_, side, :_, buffer, stencil - 1))
        elem[stencil] = :($coeff * (1 + (τ / β[$stencil])^2))
    end

    return :($(elem...),)
end

for buffer in [2, 3, 4, 5, 6]
    @eval @inline  beta_loop(ψ, ::Val{$buffer}) = @inbounds @fastmath $(metaprogrammed_beta_loop(buffer))

    for stencil in 0 : buffer - 1
        @eval @inline beta_value(ψ, ::Val{$buffer}, ::Val{$stencil}) = @inbounds @fastmath $(metaprogrammed_smoothness_sum(buffer, stencil))
    end
end

# Global smoothness indicator τ₂ᵣ₋₁ taken from "Accuracy of the weighted essentially non-oscillatory conservative finite difference schemes", Don & Borges, 2013
@inline global_smoothness_indicator(::Val{2}, β) = @inbounds abs(β[1] - β[2])
@inline global_smoothness_indicator(::Val{3}, β) = @inbounds abs(β[1] - β[3])
@inline global_smoothness_indicator(::Val{4}, β) = @inbounds abs(β[1] + 3β[2] - 3β[3] -  β[4])
@inline global_smoothness_indicator(::Val{5}, β) = @inbounds abs(β[1] + 2β[2] - 6β[3] + 2β[4] + β[5])
@inline global_smoothness_indicator(::Val{6}, β) = @inbounds abs(β[1] +  β[2] - 8β[3] + 8β[4] - β[5] - β[6])

# Calculating Dynamic WENO Weights (wᵣ), either with JS weno, Z weno or VectorInvariant WENO
for side in (:left, :right)
    biased_weno_weights = Symbol(side, :_biased_weno_weights)
    biased_β = Symbol(side, :_biased_β)
    
    tangential_stencil_u = Symbol(:tangential_, side, :_stencil_u)
    tangential_stencil_v = Symbol(:tangential_, side, :_stencil_v)
    
    for N in [2, 3, 4, 5, 6]
        @eval begin
            @inline function $biased_weno_weights(ψ, scheme::WENO{$N}, args...) 
                @inbounds begin
                    β = beta_loop(ψ, Val($N))
                    τ = global_smoothness_indicator(Val($N), β)
                    α = @fastmath $(metaprogrammed_alpha_loop(N, side))
                    return α ./ sum(α)
                end
            end

            @inline function $biased_weno_weights(ijk, scheme::WENO{$N}, dir, ::VelocityStencil, u, v) 
                @inbounds begin
                    i, j, k = ijk
                
                    ψ  = $tangential_stencil_u(i, j, k, scheme, dir, u)
                    β₁ = beta_loop(ψ, Val($N))
                    ψ  = $tangential_stencil_v(i, j, k, scheme, dir, v)
                    β₁ = beta_loop(ψ, Val($N))
                    β  = @fastmath $(metaprogrammed_beta_sum(N))
                    τ  = global_smoothness_indicator(Val($N), β)
                    α  = @fastmath $(metaprogrammed_alpha_loop(N, side))
                    return α ./ sum(α)
                end
            end
        end
    end
end

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
    return :($(stencil_full...),)
end

# Stencils for left and right biased reconstruction ((ψ̅ᵢ₋ᵣ₊ⱼ for j in 0:k) for r in 0:k) to calculate v̂ᵣ = ∑ⱼ(cᵣⱼψ̅ᵢ₋ᵣ₊ⱼ) 
# where `k = N - 1`. Coefficients (cᵣⱼ for j in 0:N) for stencil r are given by `coeff_side_p(scheme, Val(r), ...)`
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
using Oceananigans.Operators: ℑyᵃᶠᵃ, ℑxᶠᵃᵃ

@inline tangential_left_stencil_u(i, j, k, scheme, ::Val{1}, u) = @inbounds left_stencil_x(i, j, k, scheme, ℑyᵃᶠᵃ, u)
@inline tangential_left_stencil_u(i, j, k, scheme, ::Val{2}, u) = @inbounds left_stencil_y(i, j, k, scheme, ℑyᵃᶠᵃ, u)
@inline tangential_left_stencil_v(i, j, k, scheme, ::Val{1}, v) = @inbounds left_stencil_x(i, j, k, scheme, ℑxᶠᵃᵃ, v)
@inline tangential_left_stencil_v(i, j, k, scheme, ::Val{2}, v) = @inbounds left_stencil_y(i, j, k, scheme, ℑxᶠᵃᵃ, v)

@inline tangential_right_stencil_u(i, j, k, scheme, ::Val{1}, u) = @inbounds right_stencil_x(i, j, k, scheme, ℑyᵃᶠᵃ, u)
@inline tangential_right_stencil_u(i, j, k, scheme, ::Val{2}, u) = @inbounds right_stencil_y(i, j, k, scheme, ℑyᵃᶠᵃ, u)
@inline tangential_right_stencil_v(i, j, k, scheme, ::Val{1}, v) = @inbounds right_stencil_x(i, j, k, scheme, ℑxᶠᵃᵃ, v)
@inline tangential_right_stencil_v(i, j, k, scheme, ::Val{2}, v) = @inbounds right_stencil_y(i, j, k, scheme, ℑxᶠᵃᵃ, v)

# Trick to force compilation of Val(stencil-1) and avoid loops on the GPU
@inline function metaprogrammed_stencil_sum(buffer)
    elem = Vector(undef, buffer)
    for stencil = 1:buffer
        elem[stencil] = :(@inbounds w[$stencil] * func(scheme, Val($(stencil-1)), ψ[$stencil], cT, Val(val), idx, loc))
    end

    return Expr(:call, :+, elem...)
end

# Calculation of WENO reconstructed value v⋆ = ∑ᵣ(wᵣv̂ᵣ)
for buffer in [2, 3, 4, 5, 6]
    @eval begin
        @inline stencil_sum(scheme::WENO{$buffer}, ψ, w, func, cT, val, idx, loc) = @inbounds $(metaprogrammed_stencil_sum(buffer))
    end
end

# Interpolation functions
for (interp, dir, val, cT) in zip([:xᶠᵃᵃ, :yᵃᶠᵃ, :zᵃᵃᶠ], [:x, :y, :z], [1, 2, 3], [:XT, :YT, :ZT]) 
    for side in (:left, :right)
        interpolate_func = Symbol(:inner_, side, :_biased_interpolate_, interp)
        stencil          = Symbol(side, :_stencil_, dir)
        weno_weights     = Symbol(side, :_biased_weno_weights)
        biased_p         = Symbol(side, :_biased_p)
        
        @eval begin
            @inline function $interpolate_func(i, j, k, grid, 
                                               scheme::WENO{N, FT, XT, YT, ZT}, 
                                               ψ, idx, loc, args...) where {N, FT, XT, YT, ZT}
                @inbounds begin
                    ψₜ = $stencil(i, j, k, scheme, ψ, grid, args...)
                    w = $weno_weights(ψₜ, scheme, Val($val), Nothing, args...)
                    return stencil_sum(scheme, ψₜ, w, $biased_p, $cT, $val, idx, loc)
                end
            end

            @inline function $interpolate_func(i, j, k, grid, 
                                               scheme::WENO{N, FT, XT, YT, ZT}, 
                                               ψ, idx, loc, VI::AbstractSmoothnessStencil, args...) where {N, FT, XT, YT, ZT}

                @inbounds begin
                    ψₜ = $stencil(i, j, k, scheme, ψ, grid, args...)
                    w = $weno_weights(ψₜ, scheme, Val($val), VI, args...)
                    return stencil_sum(scheme, ψₜ, w, $biased_p, $cT, $val, idx, loc)
                end
            end

            @inline function $interpolate_func(i, j, k, grid, 
                                               scheme::WENO{N, FT, XT, YT, ZT}, 
                                               ψ, idx, loc, VI::VelocityStencil, u, v, args...) where {N, FT, XT, YT, ZT}

                @inbounds begin
                    ψₜ = $stencil(i, j, k, scheme, ψ, grid, u, v, args...)
                    w = $weno_weights((i, j, k), scheme, Val($val), VI, u, v)
                    return stencil_sum(scheme, ψₜ, w, $biased_p, $cT, $val, idx, loc)
                end
            end

            @inline function $interpolate_func(i, j, k, grid, 
                                               scheme::WENO{N, FT, XT, YT, ZT}, 
                                               ψ, idx, loc, VI::FunctionStencil, args...) where {N, FT, XT, YT, ZT}

                @inbounds begin
                    ψₜ = $stencil(i, j, k, scheme, ψ,       grid, args...)
                    ψₛ = $stencil(i, j, k, scheme, VI.func, grid, args...)
                    w = $weno_weights(ψₛ, scheme, Val($val), VI, args...)
                    return stencil_sum(scheme, ψₜ, w, $biased_p, $cT, $val, idx, loc)
                end
            end
        end
    end
end
