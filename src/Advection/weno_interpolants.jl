using Oceananigans.Operators: ℑyᵃᶠᵃ, ℑxᶠᵃᵃ
using Oceananigans.Utils: newton_div

include("weno_stencils.jl")
include("weno_smoothness.jl")
include("weno_coefficients.jl")

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
_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, reconstruced_function::F, bias, smoothness_stencil, args...) where F<:Function
```

For scalar reconstructions
```julia
_biased_interpolate_xᶠᵃᵃ(i, j, k, grid, bias, reconstruced_field::F) where F<:AbstractField
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

Base.show(io::IO, a::FunctionStencil) = print(io, "FunctionStencil f = $(a.func)")

const ϵ = 1f-8

# Optimal values for finite volume reconstruction of order `WENO{order}` and stencil `Val{stencil}` from
# Balsara & Shu, "Monotonicity Preserving Weighted Essentially Non-oscillatory Schemes with Inceasingly High Order of Accuracy"
for FT in fully_supported_float_types
    @eval begin
        # WENO 3rd order
        @inline C★(::WENO{2, $FT}, red_order, ::Val{0}) = ifelse(red_order==1, $(FT(1)), $(FT(2//3)))
        @inline C★(::WENO{2, $FT}, red_order, ::Val{1}) = ifelse(red_order==1, $(FT(0)), $(FT(1//3)))

        # WENO 5th order
        @inline C★(::WENO{3, $FT}, red_order, ::Val{0}) = ifelse(red_order==1, $(FT(1)), ifelse(red_order==2, $(FT(2//3)), $(FT(3//10))))
        @inline C★(::WENO{3, $FT}, red_order, ::Val{1}) = ifelse(red_order==1, $(FT(0)), ifelse(red_order==2, $(FT(1//3)), $(FT(3//5))))
        @inline C★(::WENO{3, $FT}, red_order, ::Val{2}) = ifelse(red_order <3, $(FT(0)), $(FT(1//10)))

        # WENO 7th order
        @inline C★(::WENO{4, $FT}, red_order, ::Val{0}) = ifelse(red_order==1, $(FT(1)), ifelse(red_order==2, $(FT(2//3)), ifelse(red_order==3, $(FT(3//10)), $(FT(4//35)))))
        @inline C★(::WENO{4, $FT}, red_order, ::Val{1}) = ifelse(red_order==1, $(FT(0)), ifelse(red_order==2, $(FT(1//3)), ifelse(red_order==3, $(FT(3//5)),  $(FT(18//35)))))
        @inline C★(::WENO{4, $FT}, red_order, ::Val{2}) = ifelse(red_order <3, $(FT(0)), ifelse(red_order==3, $(FT(1//10)), $(FT(12//35))))
        @inline C★(::WENO{4, $FT}, red_order, ::Val{3}) = ifelse(red_order <4, $(FT(0)), $(FT(1//35)))

        # WENO 9th order
        @inline C★(::WENO{5, $FT}, red_order, ::Val{0}) = ifelse(red_order==1, $(FT(1)), ifelse(red_order==2, $(FT(2//3)),  ifelse(red_order==3, $(FT(3//10)), ifelse(red_order==4, $(FT(4//35)),  $(FT(5//126))))))
        @inline C★(::WENO{5, $FT}, red_order, ::Val{1}) = ifelse(red_order==1, $(FT(0)), ifelse(red_order==2, $(FT(1//3)),  ifelse(red_order==3, $(FT(3//5)),  ifelse(red_order==4, $(FT(18//35)), $(FT(20//63))))))
        @inline C★(::WENO{5, $FT}, red_order, ::Val{2}) = ifelse(red_order <3, $(FT(0)), ifelse(red_order==3, $(FT(1//10)), ifelse(red_order==4, $(FT(12//35)), $(FT(100//231)))))
        @inline C★(::WENO{5, $FT}, red_order, ::Val{3}) = ifelse(red_order <4, $(FT(0)), ifelse(red_order==4, $(FT(1//35)), $(FT(10//63))))
        @inline C★(::WENO{5, $FT}, red_order, ::Val{4}) = ifelse(red_order <5, $(FT(0)), $(FT(1//126)))
    end
end

@inline function metaprogrammed_reconstruction_operation(buffer)
    elem = Vector(undef, buffer)
    for stencil = 1:buffer
        elem[stencil] = :(ψ[$stencil] * C[$(stencil)])
    end    
    return Expr(:call, :+, elem...)
end

""" 
    biased_p(ψ, scheme::WENO{buffer}, bias, ::Val{stencil})

Biased reconstruction of `ψ` from the stencil `stencil` of a WENO reconstruction of
order `buffer * 2 - 1`. The reconstruction is calculated as

```math
ψ★ = ∑ᵣ cᵣ ⋅ ψᵣ
```

where ``cᵣ`` is computed from the function `coeff_p`
"""
# Smoothness indicators for stencil `stencil` for left and right biased reconstruction
for buffer in advection_buffers[2:end] # WENO{<:Any, 1} does not exist
    @eval @inline reconstruction_operation(scheme::WENO{$buffer}, ψ, C) = @inbounds @muladd $(metaprogrammed_reconstruction_operation(buffer))
end

@inline function biased_p(ψ, scheme, red_order, val_stencil) 
    coefficients = reconstruction_coefficients(scheme, red_order, val_stencil)
    return reconstruction_operation(scheme, ψ, coefficients)
end

# The rule for calculating smoothness indicators is the following (example WENO{4} which is seventh order)
# ψ[1] (C[1]  * ψ[1] + C[2] * ψ[2] + C[3] * ψ[3] + C[4] * ψ[4]) +
# ψ[2] (C[5]  * ψ[2] + C[6] * ψ[3] + C[7] * ψ[4]) +
# ψ[3] (C[8]  * ψ[3] + C[9] * ψ[4])
# ψ[4] (C[10] * ψ[4])
# This expression is the output of metaprogrammed_smoothness_operation(4)

# Trick to force compilation of Val(stencil-1) and avoid loops on the GPU
@inline function metaprogrammed_smoothness_operation(buffer)
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

"""
    smoothness_indicator(ψ, scheme::WENO{buffer, FT}, ::Val{stencil})

Return the smoothness indicator β for the stencil number `stencil` of a WENO reconstruction of order `buffer * 2 - 1`.
The smoothness indicator (β) is calculated as follows

```julia
C = smoothness_coefficients(Val(buffer), Val(stencil))

# The smoothness indicator
β = 0
c_idx = 1
for stencil = 1:buffer - 1
    partial_sum = [C[c_idx + i - stencil)] * ψ[i]) for i in stencil:buffer]
    β          += ψ[stencil] * partial_sum
    c_idx += buffer - stencil + 1
end

β += ψ[buffer] * ψ[buffer] * C[c_idx])
```

This last operation is metaprogrammed in the function `metaprogrammed_smoothness_operation` (to avoid loops)
and, for `buffer == 3` unrolls into

```julia
β = ψ[1] * (C[1]  * ψ[1] + C[2] * ψ[2] + C[3] * ψ[3]) +
    ψ[2] * (C[4]  * ψ[2] + C[5] * ψ[3]) +
    ψ[3] * (C[6])
```

while for `buffer == 4` unrolls into

```julia
β = ψ[1] * (C[1]  * ψ[1] + C[2] * ψ[2] + C[3] * ψ[3] + C[4] * ψ[4]) +
    ψ[2] * (C[5]  * ψ[2] + C[6] * ψ[3] + C[7] * ψ[4]) +
    ψ[3] * (C[8]  * ψ[3] + C[9] * ψ[4])
    ψ[4] * (C[10] * ψ[4])
```
"""
@inline smoothness_indicator(ψ, args...) = zero(ψ[1]) # This is a fallback method, here only for documentation purposes

# Smoothness indicators for stencil `stencil` for left and right biased reconstruction
for buffer in advection_buffers[2:end] # WENO{<:Any, 1} does not exist
    @eval @inline smoothness_operation(scheme::WENO{$buffer}, ψ, C) = @inbounds @muladd $(metaprogrammed_smoothness_operation(buffer)) + ε
end

@inline function smoothness_indicator(ψ, scheme, red_order, val_stencil)
    coefficients = smoothness_coefficients(scheme, red_order, val_stencil)
    return smoothness_operation(scheme, ψ, coefficients)
end

# Shenanigans for WENO weights calculation for vector invariant formulation -> [β[i] = 0.5 * (βᵤ[i] + βᵥ[i]) for i in 1:buffer]
@inline function metaprogrammed_beta_sum(buffer)
    elem = Vector(undef, buffer)
    for stencil = 1:buffer
        elem[stencil] = :((β₁[$stencil] + β₂[$stencil])/2)
    end

    return :($(elem...),)
end

# smoothness_indicator calculation for scheme and stencil = 0:buffer - 1
@inline function metaprogrammed_beta_loop(buffer)
    elem = Vector(undef, buffer)
    for stencil = 1:buffer
        elem[stencil] = :(smoothness_indicator(ψ[$stencil], scheme, red_order, Val($(stencil-1))))
    end

    return :($(elem...),)
end

# ZWENO α weights C★ᵣ * (1 + (τ₂ᵣ₋₁ / (βᵣ + ϵ))ᵖ)
@inline function metaprogrammed_zweno_alpha_loop(buffer)
    elem = Vector(undef, buffer)
    for stencil = 1:buffer
        elem[stencil] = :(C★(scheme, red_order, Val($(stencil-1))) * (1 + (newton_div(FTS, τ, β[$stencil]))^2))
    end

    return :($(elem...),)
end

for buffer in advection_buffers[1:end]
    @eval begin
        @inline  beta_sum(scheme::WENO{$buffer, FT}, β₁, β₂)       where FT = @inbounds $(metaprogrammed_beta_sum(buffer))
        @inline beta_loop(scheme::WENO{$buffer, FT}, red_order, ψ) where FT = @inbounds $(metaprogrammed_beta_loop(buffer))
        @inline zweno_alpha_loop(scheme::WENO{$buffer, FT, FTS}, red_order, β, τ) where {FT, FTS} = @inbounds $(metaprogrammed_zweno_alpha_loop(buffer))
    end
end


"""
    function biased_weno_weights(ψ, scheme::WENO{N, FT}, red_order, args...)

Biased weno weights ω used to weight the WENO reconstruction of the different stencils.
We use here a Z-WENO formulation where

```math
    α = C★ ⋅ (1 + τ² / (β + ϵ)²)
```

where
- ``C★`` is the optimal weight that leads to an upwind reconstruction of order `N * 2 - 1`,
- ``β`` is the smoothness indicator calculated by the `smoothness_indicator` function
- ``τ`` is a global smoothness indicator, function of the ``β`` values, calculated by the `global_smoothness_indicator` function
- ``ϵ`` is a regularization constant, typically equal to 1e-8

The ``α`` values are normalized before returning
"""
@inline function biased_weno_weights(ψ, grid, scheme::WENO{N, FT}, red_order, args...) where {N, FT}
    β = beta_loop(scheme, red_order, ψ)
    τ = global_smoothness_indicator(β, red_order)
    α = zweno_alpha_loop(scheme, red_order, β, τ)
    
    # Normalization factor
    αs⁻¹ = 1 / sum(α)

    return α .* αs⁻¹
end

@inline function biased_weno_weights(ijk, grid, scheme::WENO{N, FT}, red_order, bias, dir, ::VelocityStencil, u, v) where {N, FT}
    i, j, k = ijk

    uₛ = tangential_stencil_u(i, j, k, grid, scheme, bias, dir, u)
    vₛ = tangential_stencil_v(i, j, k, grid, scheme, bias, dir, v)
    βᵤ = beta_loop(scheme, red_order, uₛ)
    βᵥ = beta_loop(scheme, red_order, vₛ)
    β  =  beta_sum(scheme, βᵤ, βᵥ)
    τ  = global_smoothness_indicator(β, red_order)
    α  = zweno_alpha_loop(scheme, red_order, β, τ)
    
    # Normalization factor
    αs⁻¹ = 1 / sum(α)

    return α .* αs⁻¹
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
                            :(ψ(i + $c, j, k, grid, args...)) :
                            dir == :y ?
                            :(ψ(i, j + $c, k, grid, args...)) :
                            :(ψ(i, j, k + $c, grid, args...))
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
for dir in (:x, :y, :z), (T, f) in zip((:Any, :Callable), (false, true))
    stencil = Symbol(:weno_stencil_, dir)
    @eval begin
        @inline function $stencil(i, j, k, grid, ::WENO{2}, bias, ψ::$T, args...)
            S = @inbounds $(load_weno_stencil(2, dir, f))
            return S₀₂(S, bias), S₁₂(S, bias)
        end

        @inline function $stencil(i, j, k, grid, ::WENO{3}, bias, ψ::$T, args...)
            S = @inbounds $(load_weno_stencil(3, dir, f))
            return S₀₃(S, bias), S₁₃(S, bias), S₂₃(S, bias)
        end

        @inline function $stencil(i, j, k, grid, ::WENO{4}, bias, ψ::$T, args...)
            S = @inbounds $(load_weno_stencil(4, dir, f))
            return S₀₄(S, bias), S₁₄(S, bias), S₂₄(S, bias), S₃₄(S, bias)
        end

        @inline function $stencil(i, j, k, grid, ::WENO{5}, bias, ψ::$T, args...)
            S = @inbounds $(load_weno_stencil(5, dir, f))
            return S₀₅(S, bias), S₁₅(S, bias), S₂₅(S, bias), S₃₅(S, bias), S₄₅(S, bias)
        end
    end
end

# Stencil for vector invariant calculation of smoothness indicators in the horizontal direction
# Parallel to the interpolation direction! (same as left/right stencil)
@inline tangential_stencil_u(i, j, k, grid, scheme, bias, ::Val{1}, u) = weno_stencil_x(i, j, k, grid, scheme, bias, ℑyᵃᶠᵃ, u)
@inline tangential_stencil_u(i, j, k, grid, scheme, bias, ::Val{2}, u) = weno_stencil_y(i, j, k, grid, scheme, bias, ℑyᵃᶠᵃ, u)
@inline tangential_stencil_v(i, j, k, grid, scheme, bias, ::Val{1}, v) = weno_stencil_x(i, j, k, grid, scheme, bias, ℑxᶠᵃᵃ, v)
@inline tangential_stencil_v(i, j, k, grid, scheme, bias, ::Val{2}, v) = weno_stencil_y(i, j, k, grid, scheme, bias, ℑxᶠᵃᵃ, v)

# Trick to force compilation of Val(stencil-1) and avoid loops on the GPU
@inline function metaprogrammed_weno_reconstruction(buffer)
    elem = Vector(undef, buffer)
    for stencil = 1:buffer
        elem[stencil] = :(ω[$stencil] * biased_p(ψ[$stencil], scheme, red_order, Val($(stencil-1))))
    end

    return Expr(:call, :+, elem...)
end

"""
    weno_reconstruction(scheme::WENO{buffer}, red_order, ψ, ω)

`bias`ed reconstruction of stencils `ψ` for a WENO scheme of order `buffer * 2 - 1` weighted by WENO
weights `ω`. `ψ` is a `Tuple` of `buffer` stencils of size `buffer` and `ω` is a `Tuple` of size `buffer`
containing the computed weights for each of the reconstruction stencils.

The calculation of the reconstruction is metaprogrammed in the `metaprogrammed_weno_reconstruction` function which, for
`buffer == 4` (seventh order WENO), unrolls to:

```julia
ψ̂ = ω[1] * biased_p(ψ[1], scheme, red_order, Val(0)) + 
    ω[2] * biased_p(ψ[2], scheme, red_order, Val(1)) + 
    ω[3] * biased_p(ψ[3], scheme, red_order, Val(2)) + 
    ω[4] * biased_p(ψ[4], scheme, red_order, Val(3)))
```

Here, [`biased_p`](@ref) is the function that computes the linear reconstruction of the individual stencils.
"""
@inline weno_reconstruction(scheme, red_order, ψ, args...) = zero(ψ[1][1]) # Fallback only for documentation purposes

# Calculation of WENO reconstructed value v⋆ = ∑ᵣ(wᵣv̂ᵣ)
for buffer in advection_buffers[2:end]
    @eval @inline weno_reconstruction(scheme::WENO{$buffer}, red_order, ψ, ω) = @inbounds @muladd $(metaprogrammed_weno_reconstruction(buffer))
end

# Interpolation functions
for (interp, dir, val) in zip([:xᶠᵃᵃ, :yᵃᶠᵃ, :zᵃᵃᶠ], [:x, :y, :z], [1, 2, 3])
    interpolate_func = Symbol(:biased_interpolate_, interp)
    stencil          = Symbol(:weno_stencil_, dir)

    @eval begin
        @inline function $interpolate_func(i, j, k, grid,
                                            scheme::WENO{N, FT}, red_order, bias,
                                            ψ, args...) where {N, FT}

            ψₜ = $stencil(i, j, k, grid, scheme, bias, ψ, args...)
            ω = biased_weno_weights(ψₜ, grid, scheme, red_order, bias, args...)
            return weno_reconstruction(scheme, red_order, ψₜ, ω)::FT
        end

        @inline function $interpolate_func(i, j, k, grid, 
                                            scheme::WENO{N, FT}, red_order, bias, 
                                            ψ, VI::AbstractSmoothnessStencil, args...) where {N, FT}

            ψₜ = $stencil(i, j, k, grid, scheme, bias, ψ, args...)
            ω = biased_weno_weights(ψₜ, grid, scheme, red_order, bias, VI, args...)
            return weno_reconstruction(scheme, red_order, ψₜ, ω)::FT
        end

        @inline function $interpolate_func(i, j, k, grid, 
                                            scheme::WENO{N, FT}, red_order, bias, 
                                            ψ, VI::VelocityStencil, u, v, args...) where {N, FT}

            ψₜ = $stencil(i, j, k, grid, scheme, bias, ψ, u, v, args...)
            ω = biased_weno_weights((i, j, k), grid, scheme, red_order, bias, Val($val), VI, u, v)
            return weno_reconstruction(scheme, red_order, ψₜ, ω)::FT
        end

        @inline function $interpolate_func(i, j, k, grid,
                                            scheme::WENO{N, FT}, red_order, bias,
                                            ψ, VI::FunctionStencil, args...) where {N, FT}

            ψₜ = $stencil(i, j, k, grid, scheme, bias, ψ,       args...)
            ψₛ = $stencil(i, j, k, grid, scheme, bias, VI.func, args...)
            ω = biased_weno_weights(ψₛ, grid, scheme, red_order, bias, VI, args...)
            return weno_reconstruction(scheme, red_order, ψₜ, ω)::FT
        end
    end
end
