using Oceananigans.Operators: ℑyᵃᶠᵃ, ℑxᶠᵃᵃ
using Oceananigans.Utils: newton_div

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
        @inline C★(::WENO{2, $FT}, ::Val{0}) = $(FT(2//3))
        @inline C★(::WENO{2, $FT}, ::Val{1}) = $(FT(1//3))

        @inline C★(::WENO{3, $FT}, ::Val{0}) = $(FT(3//10))
        @inline C★(::WENO{3, $FT}, ::Val{1}) = $(FT(3//5))
        @inline C★(::WENO{3, $FT}, ::Val{2}) = $(FT(1//10))

        @inline C★(::WENO{4, $FT}, ::Val{0}) = $(FT(4//35))
        @inline C★(::WENO{4, $FT}, ::Val{1}) = $(FT(18//35))
        @inline C★(::WENO{4, $FT}, ::Val{2}) = $(FT(12//35))
        @inline C★(::WENO{4, $FT}, ::Val{3}) = $(FT(1//35))

        @inline C★(::WENO{5, $FT}, ::Val{0}) = $(FT(5//126))
        @inline C★(::WENO{5, $FT}, ::Val{1}) = $(FT(20//63))
        @inline C★(::WENO{5, $FT}, ::Val{2}) = $(FT(10//21))
        @inline C★(::WENO{5, $FT}, ::Val{3}) = $(FT(10//63))
        @inline C★(::WENO{5, $FT}, ::Val{4}) = $(FT(1//126))

        @inline C★(::WENO{6, $FT}, ::Val{0}) = $(FT(1//77))
        @inline C★(::WENO{6, $FT}, ::Val{1}) = $(FT(25//154))
        @inline C★(::WENO{6, $FT}, ::Val{2}) = $(FT(100//231))
        @inline C★(::WENO{6, $FT}, ::Val{3}) = $(FT(25//77))
        @inline C★(::WENO{6, $FT}, ::Val{4}) = $(FT(5//77))
        @inline C★(::WENO{6, $FT}, ::Val{5}) = $(FT(1//462))
    end
end

# Reconstruction coefficients on the first differences `δ[i] = S[i+1] - S[i]`, relative to the anchor
# `S[buffer]`. The change of basis is exact because `∑ⱼ cⱼ = 1`.
function difference_reconstruction_coefficients(FT, buffer, stencil)
    c = stencil_coefficients(BigFloat, 50, stencil, collect(1:100), collect(1:100); order=buffer)
    d = zeros(BigFloat, buffer - 1)

    for k in 1:buffer - 1
        i = buffer - stencil + k - 1 # the difference δ[i] that d[k] multiplies

        for j in 1:buffer
            mⱼ = buffer - stencil + j - 1 # the stencil value S[mⱼ] that c[j] multiplies

            if i ≥ buffer && mⱼ > i
                d[k] += c[j]
            elseif i < buffer && mⱼ ≤ i
                d[k] -= c[j]
            end
        end
    end

    return FT.(tuple(d...))
end

# ENO reconstruction procedure per stencil
for buffer in advection_buffers[2:end] # WENO{<:Any, 1} does not exist
    for stencil in collect(0:1:buffer-1)
        for FT in fully_supported_float_types
            @eval begin
                """
                    coeff_p(::WENO{buffer, FT}, ::Val{stencil})

                Reconstruction coefficients for the stencil number `stencil` of a WENO reconstruction
                of order `buffer * 2 - 1`, expressed in the first differences of the stencil.
                """
                @inline coeff_p(::WENO{$buffer, $FT}, ::Val{$stencil}) =
                    @inbounds $(difference_reconstruction_coefficients(FT, buffer, stencil))
            end
        end

        @eval begin
            """
                biased_p(scheme::WENO{buffer}, ::Val{stencil}, δ)

            Reconstruction from the stencil `stencil` of a WENO reconstruction of order `buffer * 2 - 1`,
            relative to the anchor value `ψ₀`. `δ` are the `buffer - 1` first differences spanned by the
            stencil, and the reconstruction is calculated as

            ```math
            ψ★ - ψ₀ = ∑ᵢ dᵢ ⋅ δᵢ
            ```

            where ``dᵢ`` is computed from the function `coeff_p`
            """
            @inline biased_p(scheme::WENO{$buffer}, ::Val{$stencil}, δ) = @inbounds sum(coeff_p(scheme, Val($stencil)) .* δ)
        end
    end
end

# _UNIFORM_ smoothness coefficients (stretched smoothness coefficients are to be fixed!)
for FT in fully_supported_float_types
    @eval begin
        """
            smoothness_coefficients(::Val{FT}, ::Val{buffer}, ::Val{stencil})

        Return the coefficients used to calculate the smoothness indicators for the stencil
        number `stencil` of a WENO reconstruction of order `buffer * 2 - 1`. β measures the derivatives of
        the reconstructing polynomial, so it is invariant to a constant shift of the stencil and is a
        quadratic form in the `buffer - 1` first differences spanned by the stencil rather than in the
        `buffer` values themselves. The coefficients are ordered to calculate it in the following fashion:

        ```julia
        buffer  = 4
        stencil = 0

        δ = # The three differences spanned by stencil 0 with buffer 4 (7th order WENO)

        C = smoothness_coefficients(Val(buffer), Val(0))

        # The smoothness indicator
        β = δ[1] * (C[1] * δ[1] + C[2] * δ[2] + C[3] * δ[3]) +
            δ[2] * (C[4] * δ[2] + C[5] * δ[3]) +
            δ[3] * (C[6] * δ[3])
        ```

        This last operation is metaprogrammed in the function `metaprogrammed_smoothness_operation`
        """
        @inline smoothness_coefficients(::Val{$FT}, ::Val{2}, ::Val{0}) = $(FT.((1,)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{2}, ::Val{1}) = $(FT.((1,)))

        @inline smoothness_coefficients(::Val{$FT}, ::Val{3}, ::Val{0}) = $(FT.((10, -11, 4)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{3}, ::Val{1}) = $(FT.((4, -5, 4)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{3}, ::Val{2}) = $(FT.((4, -11, 10)))

        @inline smoothness_coefficients(::Val{$FT}, ::Val{4}, ::Val{0}) = $(FT.((2.107, -5.188, 1.854, 3.708, -2.788, 0.547)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{4}, ::Val{1}) = $(FT.((0.547, -1.428, 0.494, 1.468, -1.108, 0.267)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{4}, ::Val{2}) = $(FT.((0.267, -1.108, 0.494, 1.468, -1.428, 0.547)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{4}, ::Val{3}) = $(FT.((0.547, -2.788, 1.854, 3.708, -5.188, 2.107)))

        @inline smoothness_coefficients(::Val{$FT}, ::Val{5}, ::Val{0}) = $(FT.((1.07918, -4.33665, 3.25158, -0.86329, 4.7898, -7.45293, 2.01678, 2.9712, -1.63185, 0.22658)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{5}, ::Val{1}) = $(FT.((0.22658, -0.94935, 0.70218, -0.18079, 1.2513, -1.96563, 0.52158, 0.846, -0.47055, 0.06908)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{5}, ::Val{2}) = $(FT.((0.06908, -0.37185, 0.30738, -0.08209, 0.6087, -1.09413, 0.30738, 0.6087, -0.37185, 0.06908)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{5}, ::Val{3}) = $(FT.((0.06908, -0.47055, 0.52158, -0.18079, 0.846, -1.96563, 0.70218, 1.2513, -0.94935, 0.22658)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{5}, ::Val{4}) = $(FT.((0.22658, -1.63185, 2.01678, -0.86329, 2.9712, -7.45293, 3.25158, 4.7898, -4.33665, 1.07918)))

        @inline smoothness_coefficients(::Val{$FT}, ::Val{6}, ::Val{0}) = $(FT.((0.6150211, -3.5160042, 4.1046694, -2.234743, 0.471274, 5.3540984, -12.848254, 7.1025008, -1.512161, 7.8421848, -8.765266, 1.8797194, 2.4683064, -1.0645062, 0.1152561)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{6}, ::Val{1}) = $(FT.((0.1152561, -0.681287, 0.792961, -0.4254026, 0.0880548, 1.1400536, -2.7680692, 1.5189424, -0.318647, 1.7580984, -1.98067, 0.4222438, 0.5706008, -0.247217, 0.0271779)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{6}, ::Val{2}) = $(FT.((0.0271779, -0.1837242, 0.224911, -0.1213142, 0.024562, 0.3544296, -0.925294, 0.518984, -0.1079386, 0.6713736, -0.7947412, 0.1713274, 0.2534504, -0.115071, 0.0139633)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{6}, ::Val{3}) = $(FT.((0.0139633, -0.115071, 0.1713274, -0.1079386, 0.024562, 0.2534504, -0.7947412, 0.518984, -0.1213142, 0.6713736, -0.925294, 0.224911, 0.3544296, -0.1837242, 0.0271779)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{6}, ::Val{4}) = $(FT.((0.0271779, -0.247217, 0.4222438, -0.318647, 0.0880548, 0.5706008, -1.98067, 1.5189424, -0.4254026, 1.7580984, -2.7680692, 0.792961, 1.1400536, -0.681287, 0.1152561)))
        @inline smoothness_coefficients(::Val{$FT}, ::Val{6}, ::Val{5}) = $(FT.((0.1152561, -1.0645062, 1.8797194, -1.512161, 0.471274, 2.4683064, -8.765266, 7.1025008, -2.234743, 7.8421848, -12.848254, 4.1046694, 5.3540984, -3.5160042, 0.6150211)))
    end
end

# The rule for calculating smoothness indicators is the following (example WENO{4} which is seventh order),
# where δ are the three differences spanned by the stencil
# δ[1] (C[1] * δ[1] + C[2] * δ[2] + C[3] * δ[3]) +
# δ[2] (C[4] * δ[2] + C[5] * δ[3]) +
# δ[3] (C[6] * δ[3])
# This expression is the output of metaprogrammed_smoothness_operation(4)

# Trick to force compilation of Val(stencil-1) and avoid loops on the GPU
@inline function metaprogrammed_smoothness_operation(buffer)
    N = buffer - 1

    elem = Vector{Expr}(undef, N)
    c_idx = 1

    for stencil = 1:N - 1
        local c = c_idx # Avoid capturing `c_idx` in the generator expression below
        stencil_sum   = Expr(:call, :+, (:(C[$(c + i - stencil)] * δ[$i]) for i in stencil:N)...)
        elem[stencil] = :(δ[$stencil] * $stencil_sum)
        c_idx += N - stencil + 1
    end

    elem[N] = :(δ[$N] * δ[$N] * C[$c_idx])

    return Expr(:call, :+, elem...)
end

"""
$(TYPEDSIGNATURES)

Return the smoothness indicator β for the stencil number `stencil` of a WENO reconstruction of order `buffer * 2 - 1`,
from the `buffer - 1` first differences `δ` spanned by that stencil.
The smoothness indicator (β) is calculated as follows

```julia
C = smoothness_coefficients(Val(buffer), Val(stencil))

# The smoothness indicator
β = 0
c_idx = 1
for stencil = 1:buffer - 2
    partial_sum = [C[c_idx + i - stencil)] * δ[i]) for i in stencil:buffer-1]
    β          += δ[stencil] * partial_sum
    c_idx += buffer - stencil
end

β += δ[buffer-1] * δ[buffer-1] * C[c_idx])
```

This last operation is metaprogrammed in the function `metaprogrammed_smoothness_operation` (to avoid loops)
and, for `buffer == 3` unrolls into

```julia
β = δ[1] * (C[1] * δ[1] + C[2] * δ[2]) +
    δ[2] * (C[3] * δ[2])
```

while for `buffer == 4` unrolls into

```julia
β = δ[1] * (C[1] * δ[1] + C[2] * δ[2] + C[3] * δ[3]) +
    δ[2] * (C[4] * δ[2] + C[5] * δ[3]) +
    δ[3] * (C[6] * δ[3])
```
"""
@inline smoothness_indicator(δ, args...) = zero(δ[1]) # This is a fallback method, here only for documentation purposes

# Smoothness indicators for stencil `stencil` for left and right biased reconstruction
for buffer in advection_buffers[2:end] # WENO{<:Any, 1} does not exist
    @eval @inline smoothness_operation(scheme::WENO{$buffer}, δ, C) = @inbounds @muladd $(metaprogrammed_smoothness_operation(buffer))

    for stencil in 0:buffer-1, FT in fully_supported_float_types
        @eval @inline smoothness_indicator(δ, scheme::WENO{$buffer, $FT}, ::Val{$stencil}) =
                      smoothness_operation(scheme, δ, $(smoothness_coefficients(Val(FT), Val(buffer), Val(stencil))))
    end
end

# Shenanigans for WENO weights calculation for vector invariant formulation -> [β[i] = 0.5 * (βᵤ[i] + βᵥ[i]) for i in 1:buffer]
@inline function metaprogrammed_beta_sum(buffer)
    elem = Vector(undef, buffer)
    for stencil = 1:buffer
        elem[stencil] = :((β₁[$stencil] + β₂[$stencil])/2)
    end

    return :($(elem...),)
end

# The `buffer - 1` differences spanned by stencil number `stencil`
stencil_differences(buffer, stencil) = Expr(:tuple, (:(δ[$i]) for i in (buffer - stencil):(2buffer - 2 - stencil))...)

# smoothness_indicator calculation for scheme and stencil = 0:buffer - 1
@inline function metaprogrammed_beta_loop(buffer)
    elem = Vector(undef, buffer)
    for stencil = 1:buffer
        elem[stencil] = :(smoothness_indicator($(stencil_differences(buffer, stencil-1)), scheme, Val($(stencil-1))))
    end

    return :($(elem...),)
end

# ZWENO α weights C★ᵣ * (1 + (τ₂ᵣ₋₁ / (βᵣ + ϵ))ᵖ)
@inline function metaprogrammed_zweno_alpha_loop(buffer)
    elem = Vector(undef, buffer)
    for stencil = 1:buffer
        elem[stencil] = :(C★(scheme, Val($(stencil-1))) * (1 + (newton_div(WCT, τ, β[$stencil] + ϵ))^2))
    end

    return :($(elem...),)
end

for buffer in advection_buffers[2:end]
    @eval begin
        @inline         beta_sum(scheme::WENO{$buffer, FT}, β₁, β₂)    where FT = @inbounds $(metaprogrammed_beta_sum(buffer))
        @inline        beta_loop(scheme::WENO{$buffer, FT}, δ)         where FT = @inbounds $(metaprogrammed_beta_loop(buffer))
        @inline zweno_alpha_loop(scheme::WENO{$buffer, FT, WCT}, β, τ) where {FT, WCT} = @inbounds $(metaprogrammed_zweno_alpha_loop(buffer))
    end
end

# Global smoothness indicator τ₂ᵣ₋₁ from "Accuracy of the weighted essentially non-oscillatory conservative finite difference schemes", Don & Borges, 2013
@inline global_smoothness_indicator(::Val{2}, β) = @inbounds abs(β[1] - β[2])
@inline global_smoothness_indicator(::Val{3}, β) = @inbounds abs(β[1] - β[3])
@inline global_smoothness_indicator(::Val{4}, β) = @inbounds abs(β[1] +  3β[2] -   3β[3] -    β[4])
@inline global_smoothness_indicator(::Val{5}, β) = @inbounds abs(β[1] +  2β[2] -   6β[3] +   2β[4] + β[5])
@inline global_smoothness_indicator(::Val{6}, β) = @inbounds abs(β[1] + 36β[2] + 135β[3] - 135β[4] - 36β[5] - β[6])

"""
$(TYPEDSIGNATURES)

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
@inline function biased_weno_weights(δ, grid, scheme::WENO{N, FT}, args...) where {N, FT}
    β = beta_loop(scheme, δ)
    τ = global_smoothness_indicator(Val(N), β)
    α = zweno_alpha_loop(scheme, β, τ)
    Σα⁻¹ =  1 / sum(α)
    return α .* Σα⁻¹
end

@inline function biased_weno_weights(ijk, grid, scheme::WENO{N, FT}, bias, dir, ::VelocityStencil, u, v) where {N, FT}
    i, j, k = ijk

    uₛ = tangential_stencil_u(i, j, k, grid, scheme, bias, dir, u)
    vₛ = tangential_stencil_v(i, j, k, grid, scheme, bias, dir, v)
    βᵤ = beta_loop(scheme, weno_differences(scheme, uₛ))
    βᵥ = beta_loop(scheme, weno_differences(scheme, vₛ))
    β  = beta_sum(scheme, βᵤ, βᵥ)

    τ = global_smoothness_indicator(Val(N), β)
    α = zweno_alpha_loop(scheme, β, τ)
    Σα⁻¹ =  1 / sum(α)
    return α .* Σα⁻¹
end

"""
    load_weno_stencil(buffer, dir, func::Bool = false)

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
```
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

# The right-biased stencil is the mirror of the left-biased one, so both use the left-biased coefficients
for dir in (:x, :y, :z), (T, f) in zip((:Any, :Function), (false, true))
    stencil = Symbol(:weno_stencil_, dir)
    @eval begin
        @inline function $stencil(i, j, k, grid, ::WENO{2}, bias, ψ::$T, args...)
            S = @inbounds $(load_weno_stencil(2, dir, f))
            return @inbounds ifelse(bias == LeftBias, (S[1], S[2], S[3]), (S[4], S[3], S[2]))
        end

        @inline function $stencil(i, j, k, grid, ::WENO{3}, bias, ψ::$T, args...)
            S = @inbounds $(load_weno_stencil(3, dir, f))
            return @inbounds ifelse(bias == LeftBias, (S[1], S[2], S[3], S[4], S[5]), (S[6], S[5], S[4], S[3], S[2]))
        end

        @inline function $stencil(i, j, k, grid, ::WENO{4}, bias, ψ::$T, args...)
            S = @inbounds $(load_weno_stencil(4, dir, f))
            return @inbounds ifelse(bias == LeftBias, (S[1], S[2], S[3], S[4], S[5], S[6], S[7]), (S[8], S[7], S[6], S[5], S[4], S[3], S[2]))
        end

        @inline function $stencil(i, j, k, grid, ::WENO{5}, bias, ψ::$T, args...)
            S = @inbounds $(load_weno_stencil(5, dir, f))
            return @inbounds ifelse(bias == LeftBias, (S[1], S[2], S[3], S[4], S[5], S[6], S[7], S[8], S[9]), (S[10], S[9], S[8], S[7], S[6], S[5], S[4], S[3], S[2]))
        end

        @inline function $stencil(i, j, k, grid, ::WENO{6}, bias, ψ::$T, args...)
            S = @inbounds $(load_weno_stencil(6, dir, f))
            return @inbounds ifelse(bias == LeftBias, (S[1], S[2], S[3], S[4], S[5], S[6], S[7], S[8], S[9], S[10], S[11]), (S[12], S[11], S[10], S[9], S[8], S[7], S[6], S[5], S[4], S[3], S[2]))
        end
    end
end

@inline weno_anchor(::WENO{N}, S) where N = @inbounds S[N]
@inline weno_differences(::WENO{N}, S) where N = @inbounds ntuple(i -> S[i+1] - S[i], Val(2N - 2))

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
        elem[stencil] = :(ω[$stencil] * biased_p(scheme, Val($(stencil-1)), $(stencil_differences(buffer, stencil-1))))
    end

    return Expr(:call, :+, :ψ₀, elem...)
end

"""
$(TYPEDSIGNATURES)

Reconstruction of a WENO scheme of order `buffer * 2 - 1` from the anchor value `ψ₀` and the `2buffer - 2`
first differences `δ` of the bias-ordered stencil, weighted by the WENO weights `ω`.

The calculation of the reconstruction is metaprogrammed in the `metaprogrammed_weno_reconstruction` function which, for
`buffer == 4` (seventh order WENO), unrolls to:

```julia
ψ̂ = ψ₀ + ω[1] * biased_p(scheme, Val(0), (δ[4], δ[5], δ[6])) +
         ω[2] * biased_p(scheme, Val(1), (δ[3], δ[4], δ[5])) +
         ω[3] * biased_p(scheme, Val(2), (δ[2], δ[3], δ[4])) +
         ω[4] * biased_p(scheme, Val(3), (δ[1], δ[2], δ[3]))
```

Here, [`biased_p`](@ref) is the function that computes the linear reconstruction of the individual stencils.
"""
@inline weno_reconstruction(scheme, ψ₀, δ, args...) = ψ₀ # Fallback only for documentation purposes

# Calculation of WENO reconstructed value v⋆ = ∑ᵣ(wᵣv̂ᵣ)
for buffer in advection_buffers[2:end]
    @eval @inline weno_reconstruction(scheme::WENO{$buffer}, ψ₀, δ, ω) = @inbounds @muladd $(metaprogrammed_weno_reconstruction(buffer))
end

# Interpolation functions
for (interp, dir, val) in zip([:xᶠᵃᵃ, :yᵃᶠᵃ, :zᵃᵃᶠ], [:x, :y, :z], [1, 2, 3])
    interpolate_func = Symbol(:biased_interpolate_, interp)
    stencil          = Symbol(:weno_stencil_, dir)

    @eval begin
        @inline function $interpolate_func(i, j, k, grid,
                                            scheme::WENO{N, FT}, bias,
                                            ψ, args...) where {N, FT}

            S  = $stencil(i, j, k, grid, scheme, bias, ψ, args...)
            ψ₀ = weno_anchor(scheme, S)
            δ  = weno_differences(scheme, S)
            ω  = biased_weno_weights(δ, grid, scheme, bias, args...)
            return weno_reconstruction(scheme, ψ₀, δ, ω)
        end

        @inline function $interpolate_func(i, j, k, grid,
                                            scheme::WENO{N, FT}, bias,
                                            ψ, VI::AbstractSmoothnessStencil, args...) where {N, FT}

            S  = $stencil(i, j, k, grid, scheme, bias, ψ, args...)
            ψ₀ = weno_anchor(scheme, S)
            δ  = weno_differences(scheme, S)
            ω  = biased_weno_weights(δ, grid, scheme, bias, VI, args...)
            return weno_reconstruction(scheme, ψ₀, δ, ω)
        end

        @inline function $interpolate_func(i, j, k, grid,
                                            scheme::WENO{N, FT}, bias,
                                            ψ, VI::VelocityStencil, u, v, args...) where {N, FT}

            S  = $stencil(i, j, k, grid, scheme, bias, ψ, u, v, args...)
            ψ₀ = weno_anchor(scheme, S)
            δ  = weno_differences(scheme, S)
            ω  = biased_weno_weights((i, j, k), grid, scheme, bias, Val($val), VI, u, v)
            return weno_reconstruction(scheme, ψ₀, δ, ω)
        end

        @inline function $interpolate_func(i, j, k, grid,
                                            scheme::WENO{N, FT}, bias,
                                            ψ, VI::FunctionStencil, args...) where {N, FT}

            S  = $stencil(i, j, k, grid, scheme, bias, ψ, args...)
            ψ₀ = weno_anchor(scheme, S)
            δ  = weno_differences(scheme, S)
            Sₛ = $stencil(i, j, k, grid, scheme, bias, VI.func, args...)
            ω  = biased_weno_weights(weno_differences(scheme, Sₛ), grid, scheme, bias, VI, args...)
            return weno_reconstruction(scheme, ψ₀, δ, ω)
        end
    end
end
