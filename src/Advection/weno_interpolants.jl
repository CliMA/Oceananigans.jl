using Oceananigans.Operators: ℑyᵃᶠᵃ, ℑxᶠᵃᵃ
using Oceananigans.Utils: newton_div
using BFloat16s: BFloat16

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
biased_interpolate_xᶠᶜᶜ(i, j, k, grid, reconstruced_function::F, bias, smoothness_stencil, args...) where F<:Function
```

For scalar reconstructions
```julia
biased_interpolate_xᶠᶜᶜ(i, j, k, grid, bias, reconstruced_field::F) where F<:AbstractField
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

@inline C★(scheme, stencil::Val, ::Nothing) = C★(scheme, stencil)

# Reconstruction coefficients on the first differences `δ[i] = S[i+1] - S[i]`, relative to the anchor
# `S[buffer]`, from the coefficients `c` of the cell values. The change of basis is exact because `∑ⱼ cⱼ = 1`.
function difference_coefficients(c, buffer, stencil)
    d = zeros(eltype(c), buffer - 1)

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

    return d
end

difference_reconstruction_coefficients(FT, buffer, stencil) =
    FT.(tuple(difference_coefficients(collect(stencil_coefficients(BigFloat, 50, stencil, collect(1:100), collect(1:100); order=buffer)), buffer, stencil)...))

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
            @inline biased_p(scheme::WENO{$buffer}, ::Val{$stencil}, δ, coefficients = nothing) =
                @inbounds @muladd $(Expr(:call, :+, (:(coeff_p(scheme, Val($stencil), coefficients)[$i] * δ[$i]) for i in 1:buffer-1)...))
        end
    end
end

@inline coeff_p(scheme, stencil::Val, ::Nothing) = coeff_p(scheme, stencil)

#####
##### Smoothness indicators
#####

# Normalization of the smoothness indicators tabulated in the literature relative to the Jiang & Shu definition,
# for `buffer = 2:6` (orders 3, 5, 7, 9, 11). The orders 7 and 9 follow Balsara & Shu (2000).
const reference_smoothness_normalization = (1, 3, 6//25, 63//1250, 189//15625)

"""
    exact_smoothness_matrix(buffer, stencil)

Return the `buffer × buffer` matrix `Q` of the Jiang & Shu smoothness indicator

```math
β = ∑ₗ₌₁ᵇᵘᶠᶠᵉʳ⁻¹ ∫ (dˡpᵣ / dxˡ)² dx = ψᵀ Q ψ
```

of the stencil number `stencil` of a WENO reconstruction of order `2buffer - 1`, computed exactly in rational
arithmetic. `pᵣ` is the polynomial that reconstructs the cell averages `ψ` of the stencil (ordered from left to
right), the integral spans the cell whose right face is reconstructed (of unit width), and `Q` is multiplied by
the normalization of the reference tables.
"""
exact_smoothness_matrix(buffer, stencil) =
    smoothness_matrix([Rational{BigInt}(m - (stencil + 1)) - 1//2 for m in 1:buffer+1]) * reference_smoothness_normalization[buffer - 1]

# The Jiang & Shu quadratic form of the cells between consecutive `edges`, in units of the reconstructed cell [-1/2, 1/2]
function smoothness_matrix(edges)
    R = eltype(edges)
    k = length(edges) - 1

    # Cell averages of the monomials xⁿ over the cells
    A = zeros(R, k, k)
    for m in 1:k, n in 0:k-1
        A[m, n+1] = (edges[m+1]^(n+1) - edges[m]^(n+1)) / ((n + 1) * (edges[m+1] - edges[m]))
    end
    B = inv(A) # polynomial coefficients aₙ = ∑ⱼ B[n+1, j] ψⱼ

    # Integrals ∫ xᵃ xᵇ dx over the reconstructed cell [-1/2, 1/2]
    M = zeros(R, k, k)
    for a in 0:k-1, b in 0:k-1
        M[a+1, b+1] = iseven(a + b) ? R(2, (a + b + 1) * BigInt(2)^(a + b + 1)) : zero(R)
    end

    Q = zeros(R, k, k)
    for l in 1:k-1
        Dₗ = zeros(R, k, k) # coefficients of the l-th derivative, xⁿ → n! / (n - l)! xⁿ⁻ˡ
        for n in l:k-1, j in 1:k
            Dₗ[n-l+1, j] = B[n+1, j] * prod(n-l+1:n)
        end
        Q += Dₗ' * M * Dₗ
    end

    return Q
end

# LDLᵀ factorization of the quadratic form β = δᵀ Qδ δ, packed as, for each l, the strictly-lower column of L
# followed by Dₗ, so that β = ∑ₗ Dₗ (δₗ + ∑ᵢ₌ₗ₊₁ Lᵢₗ δᵢ)²
function ldlt_smoothness_coefficients(Qδ)
    N = size(Qδ, 1)
    R = eltype(Qδ)
    L = zeros(R, N, N)
    D = zeros(R, N)

    for j in 1:N
        L[j, j] = 1
        D[j] = Qδ[j, j] - sum((L[j, k]^2 * D[k] for k in 1:j-1); init=zero(R))
        for i in j+1:N
            L[i, j] = (Qδ[i, j] - sum((L[i, k] * L[j, k] * D[k] for k in 1:j-1); init=zero(R))) / D[j]
        end
    end

    C = R[]
    for l in 1:N
        append!(C, L[l+1:N, l])
        push!(C, D[l])
    end

    return Tuple(C)
end

"""
    smoothness_coefficients(::Val{FT}, ::Val{buffer}, ::Val{stencil})

Return the coefficients used to calculate the smoothness indicator for the stencil number `stencil`
of a WENO reconstruction of order `buffer * 2 - 1`: the packed LDLᵀ factorization of the smoothness
indicator as a quadratic form in the `buffer - 1` first differences `δ` spanned by the stencil, computed
exactly and rounded to `FT`. They are ordered to calculate β in the following fashion:

```julia
buffer  = 4
stencil = 0

δ = # The three differences spanned by stencil 0 with buffer 4 (7th order WENO)

C = smoothness_coefficients(Val(Float64), Val(buffer), Val(stencil))

# The smoothness indicator
β = C[3] * (δ[1] + C[1] * δ[2] + C[2] * δ[3])^2 +
    C[5] * (δ[2] + C[4] * δ[3])^2 +
    C[6] *  δ[3]^2
```

This last operation is metaprogrammed in the function `metaprogrammed_smoothness_operation`
"""
@inline smoothness_coefficients(::Val{FT}, ::Val{buffer}, ::Val{stencil}) where {FT, buffer, stencil} = nothing # Fallback only for documentation purposes

for buffer in advection_buffers[2:end], stencil in 0:buffer-1
    Q = exact_smoothness_matrix(buffer, stencil)

    # β is invariant to a constant shift of the stencil, so substituting ψₘ = ψ₁ + ∑ᵢ₌₁ᵐ⁻¹ δᵢ = ψ₁ + (T δ)ₘ
    # makes it a quadratic form in the first differences δ
    T = [Int(i < m) for m in 1:buffer, i in 1:buffer-1]
    C = ldlt_smoothness_coefficients(T' * Q * T)

    for FT in fully_supported_float_types
        @eval @inline smoothness_coefficients(::Val{$FT}, ::Val{$buffer}, ::Val{$stencil}) = $(FT.(C))
    end
end

# The rule for calculating smoothness indicators is the following (example WENO{4} which is seventh order),
# where δ are the three differences spanned by the stencil
# C[3] * (δ[1] + C[1] * δ[2] + C[2] * δ[3])^2 +
# C[5] * (δ[2] + C[4] * δ[3])^2 +
# C[6] *  δ[3]^2
# This expression is the output of metaprogrammed_smoothness_operation(4)

# Trick to force compilation of Val(stencil-1) and avoid loops on the GPU
@inline function metaprogrammed_smoothness_operation(buffer)
    N = buffer - 1
    elem = Vector{Expr}(undef, N)
    p = 1
    for l in 1:N
        q = p # Avoid boxing
        t = Expr(:call, :+, :(δ[$l]), (:(C[$(q + i - l - 1)] * δ[$i]) for i in l+1:N)...)
        elem[l] = :(C[$(q + N - l)] * $t * $t)
        p += N - l + 1
    end
    return Expr(:call, :+, elem...)
end

"""
$(TYPEDSIGNATURES)

Return the smoothness indicator β for the stencil number `stencil` of a WENO reconstruction of order `buffer * 2 - 1`,
from the `buffer - 1` first differences `δ` spanned by that stencil. β is evaluated as a sum of squares from the packed
LDLᵀ coefficients `C = smoothness_coefficients(Val(FT), Val(buffer), Val(stencil))`:

```julia
N = buffer - 1
β = 0
p = 1
for l = 1:N
    t  = δ[l] + sum(C[p + i - l - 1] * δ[i] for i in l+1:N)
    β += C[p + N - l] * t^2
    p += N - l + 1
end
```

This operation is metaprogrammed in the function `metaprogrammed_smoothness_operation` (to avoid loops)
and, for `buffer == 3` unrolls into

```julia
β = C[2] * (δ[1] + C[1] * δ[2])^2 + C[3] * δ[2]^2
```

while for `buffer == 4` unrolls into

```julia
β = C[3] * (δ[1] + C[1] * δ[2] + C[2] * δ[3])^2 +
    C[5] * (δ[2] + C[4] * δ[3])^2 +
    C[6] *  δ[3]^2
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

@inline smoothness_indicator(δ, scheme, stencil::Val, ::Nothing) = smoothness_indicator(δ, scheme, stencil)

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
        elem[stencil] = :(smoothness_indicator($(stencil_differences(buffer, stencil-1)), scheme, Val($(stencil-1)), coefficients))
    end

    return :($(elem...),)
end

# ZWENO α weights C★ᵣ * (1 + (τ₂ᵣ₋₁ / (βᵣ + ϵ★))ᵖ), with the regularization ϵ★ = zweno_regularization(scheme, τ)
@inline function metaprogrammed_zweno_alpha_loop(buffer)
    elem = Vector(undef, buffer)
    for stencil = 1:buffer
        elem[stencil] = :(C★(scheme, Val($(stencil-1)), coefficients) * zweno_ratio_term(newton_div(WCT, τ, β[$stencil] + ϵ★)))
    end

    return quote
        ϵ★ = zweno_regularization(scheme, τ)
        ($(elem...),)
    end
end

# 1 + r², fused
@inline zweno_ratio_term(r) = muladd(r, r, 1)

@inline zweno_regularization(scheme, τ) = oftype(τ, ϵ)

for FT in (Float32, BFloat16)
    @eval @inline zweno_regularization(::WENO{N, $FT}, τ) where N = max(oftype(τ, ϵ), τ * $(inv(FT(2)^62)))
end

for buffer in advection_buffers[2:end]
    @eval begin
        @inline         beta_sum(scheme::WENO{$buffer, FT}, β₁, β₂)    where FT = @inbounds $(metaprogrammed_beta_sum(buffer))
        @inline        beta_loop(scheme::WENO{$buffer, FT}, δ, coefficients = nothing)         where FT = @inbounds $(metaprogrammed_beta_loop(buffer))
        @inline zweno_alpha_loop(scheme::WENO{$buffer, FT, WCT}, β, τ, coefficients = nothing) where {FT, WCT} = @inbounds $(metaprogrammed_zweno_alpha_loop(buffer))
    end
end

# Global smoothness indicator τ₂ᵣ₋₁ from "Accuracy of the weighted essentially non-oscillatory conservative finite difference schemes", Don & Borges, 2013
@inline global_smoothness_indicator(::Val{2}, β) = @inbounds abs(β[1] - β[2])
@inline global_smoothness_indicator(::Val{3}, β) = @inbounds abs(β[1] - β[3])
@inline global_smoothness_indicator(::Val{4}, β) = @inbounds @muladd abs(β[1] +  3β[2] -   3β[3] -    β[4])
@inline global_smoothness_indicator(::Val{5}, β) = @inbounds @muladd abs(β[1] +  2β[2] -   6β[3] +   2β[4] + β[5])
@inline global_smoothness_indicator(::Val{6}, β) = @inbounds @muladd abs(β[1] + 36β[2] + 135β[3] - 135β[4] - 36β[5] - β[6])

# 1 / ∑α. Ordinary division by default: the approximate reciprocal slows the Float64 tracer kernels on CUDA.
@inline function weight_normalization(scheme, α)
    Σα = sum(α)
    return one(Σα) / Σα
end

# In Float32 and BFloat16, the division method of the weight computation is faster. ∑α ≲ 2¹²⁴
# (see `zweno_regularization`), so its approximate reciprocal cannot flush to zero.
for FT in (Float32, BFloat16)
    @eval @inline function weight_normalization(::WENO{N, $FT, WCT}, α) where {N, WCT}
        Σα = sum(α)
        return newton_div(WCT, one(Σα), Σα)
    end
end

"""
$(TYPEDSIGNATURES)

Biased weno weights ω used to weight the WENO reconstruction of the different stencils.
We use here a Z-WENO formulation where

```math
    α = C★ ⋅ (1 + τ² / (β + ϵ★)²)
```

where
- ``C★`` is the optimal weight that leads to an upwind reconstruction of order `N * 2 - 1`,
- ``β`` is the smoothness indicator calculated by the `smoothness_indicator` function
- ``τ`` is a global smoothness indicator, function of the ``β`` values, calculated by the `global_smoothness_indicator` function
- ``ϵ★`` is the regularization computed by `zweno_regularization`: the constant ``ϵ = 10⁻⁸``, which grows with
  ``τ`` beside large jumps in Float32 and BFloat16 so that ``α`` cannot overflow

The ``α`` values are returned unnormalized: [`weno_reconstruction`](@ref) divides by their sum.
"""
@inline function biased_weno_weights(δ, scheme::WENO{N, FT}, coefficients = nothing) where {N, FT}
    β = beta_loop(scheme, δ, coefficients)
    τ = global_smoothness_indicator(Val(N), β)
    return zweno_alpha_loop(scheme, β, τ, coefficients)
end

# The right-biased stencil is the mirror of the left-biased one, so both use the left-biased coefficients
@inline weno_stencil(::WENO{N}, bias, S) where N = @inbounds ifelse(bias == LeftBias, ntuple(n -> S[n], Val(2N - 1)), ntuple(n -> S[2N + 1 - n], Val(2N - 1)))

@inline weno_anchor(::WENO{N}, S) where N = @inbounds S[N]
@inline weno_differences(::WENO{N}, S) where N = @inbounds ntuple(i -> S[i+1] - S[i], Val(2N - 2))

# Trick to force compilation of Val(stencil-1) and avoid loops on the GPU
@inline function metaprogrammed_weno_reconstruction(buffer)
    elem = Vector(undef, buffer)
    for stencil = 1:buffer
        elem[stencil] = :(ω[$stencil] * biased_p(scheme, Val($(stencil-1)), $(stencil_differences(buffer, stencil-1)), coefficients))
    end

    return :(ψ₀ + $(Expr(:call, :+, elem...)) * weight_normalization(scheme, ω))
end

"""
$(TYPEDSIGNATURES)

Reconstruction of a WENO scheme of order `buffer * 2 - 1` from the anchor value `ψ₀` and the `2buffer - 2`
first differences `δ` of the bias-ordered stencil, weighted by the unnormalized WENO weights `ω`.

The calculation of the reconstruction is metaprogrammed in the `metaprogrammed_weno_reconstruction` function which, for
`buffer == 4` (seventh order WENO), unrolls to:

```julia
ψ̂ = ψ₀ + (ω[1] * biased_p(scheme, Val(0), (δ[4], δ[5], δ[6])) +
          ω[2] * biased_p(scheme, Val(1), (δ[3], δ[4], δ[5])) +
          ω[3] * biased_p(scheme, Val(2), (δ[2], δ[3], δ[4])) +
          ω[4] * biased_p(scheme, Val(3), (δ[1], δ[2], δ[3]))) / (ω[1] + ω[2] + ω[3] + ω[4])
```

Here, [`biased_p`](@ref) is the function that computes the linear reconstruction of the individual stencils.
"""
@inline weno_reconstruction(scheme, ψ₀, δ, args...) = ψ₀ # Fallback only for documentation purposes

# Calculation of WENO reconstructed value v⋆ = ∑ᵣ(wᵣv̂ᵣ) / ∑ᵣwᵣ
for buffer in advection_buffers[2:end]
    @eval @inline weno_reconstruction(scheme::WENO{$buffer}, ψ₀, δ, ω, coefficients = nothing) = @inbounds @muladd $(metaprogrammed_weno_reconstruction(buffer))
end

@inline weno_coefficients(::Nothing, args...) = nothing

@inline coefficient_direction(::Val{1}) = Val(:x)
@inline coefficient_direction(::Val{2}) = Val(:y)
@inline coefficient_direction(::Val{3}) = Val(:z)

@inline coefficient_location(::Face)   = Val(:faces)
@inline coefficient_location(::Center) = Val(:centers)

# The stretched coefficients of the point `(i, j, k)` located at `ℓ` and reconstructed along `dir`, `nothing` on uniform directions
@inline stretched_coefficients(i, j, k, dir::Val{d}, ℓ, scheme::WENO, bias) where d =
    weno_coefficients(scheme.coefficients, coefficient_direction(dir), coefficient_location(ℓ[d]), (i, j, k)[d], bias)

# Reconstruction from the `2N` values `S` of `load_stencil`, with the stretched coefficients `C`
@inline function biased_interpolate(scheme::WENO{N, FT}, bias, S, C = nothing) where {N, FT}
    S  = weno_stencil(scheme, bias, S)
    ψ₀ = weno_anchor(scheme, S)
    δ  = weno_differences(scheme, S)
    ω  = biased_weno_weights(δ, scheme, C)
    return weno_reconstruction(scheme, ψ₀, δ, ω, C)
end

@inline stretched_interpolate(scheme::WENO, bias, S, C) = biased_interpolate(scheme, bias, S, C)

# the reconstruction with the optimal weights, which is the upwind-biased reconstruction of the same order
@inline linear_interpolate(::WENO{N, FT}, bias, S) where {N, FT} = biased_interpolate(UpwindBiased{N, FT}(nothing, nothing, nothing), bias, S)

# Interpolation functions
@inline function biased_interpolate(i, j, k, grid, dir, ℓ, scheme::WENO{N, FT}, bias, ψ, args...) where {N, FT}
    S = load_stencil(i, j, k, grid, dir, ℓ, Val(N), ψ, args...)
    C = stretched_coefficients(i, j, k, dir, ℓ, scheme, bias)
    return biased_interpolate(scheme, bias, S, C)
end

@inline function biased_interpolate(i, j, k, grid, dir, ℓ, scheme::WENO{N, FT}, bias, ψ, VI::AbstractSmoothnessStencil, args...) where {N, FT}
    S = load_stencil(i, j, k, grid, dir, ℓ, Val(N), ψ, args...)
    C = stretched_coefficients(i, j, k, dir, ℓ, scheme, bias)
    return biased_interpolate(scheme, bias, S, C)
end

# The smoothness stencils continue past walls with the agreement factors of the reconstructed stencil
@inline function biased_interpolate(i, j, k, grid, dir, ℓ, scheme::WENO{N, FT}, bias, ψ, VI::VelocityStencil, u, v, args...) where {N, FT}
    S,  G = load_stencil(i, j, k, grid, dir, ℓ, Val(N), nothing, ψ, u, v, args...)
    Sᵤ, _ = load_stencil(i, j, k, grid, dir, ℓ, Val(N), G, ℑyᵃᶠᵃ, u)
    Sᵥ, _ = load_stencil(i, j, k, grid, dir, ℓ, Val(N), G, ℑxᶠᵃᵃ, v)
    C  = stretched_coefficients(i, j, k, dir, ℓ, scheme, bias)
    S  = weno_stencil(scheme, bias, S)
    Sᵤ = weno_stencil(scheme, bias, Sᵤ)
    Sᵥ = weno_stencil(scheme, bias, Sᵥ)
    ψ₀ = weno_anchor(scheme, S)
    δ  = weno_differences(scheme, S)
    δᵤ = weno_differences(scheme, Sᵤ)
    δᵥ = weno_differences(scheme, Sᵥ)
    βᵤ = beta_loop(scheme, δᵤ, C)
    βᵥ = beta_loop(scheme, δᵥ, C)
    β  = beta_sum(scheme, βᵤ, βᵥ)
    τ  = global_smoothness_indicator(Val(N), β)
    ω  = zweno_alpha_loop(scheme, β, τ, C)
    return weno_reconstruction(scheme, ψ₀, δ, ω, C)
end

@inline function biased_interpolate(i, j, k, grid, dir, ℓ, scheme::WENO{N, FT}, bias, ψ, VI::FunctionStencil, args...) where {N, FT}
    S,  G = load_stencil(i, j, k, grid, dir, ℓ, Val(N), nothing, ψ, args...)
    Sₛ, _ = load_stencil(i, j, k, grid, dir, ℓ, Val(N), G, VI.func, args...)
    C  = stretched_coefficients(i, j, k, dir, ℓ, scheme, bias)
    S  = weno_stencil(scheme, bias, S)
    Sₛ = weno_stencil(scheme, bias, Sₛ)
    ψ₀ = weno_anchor(scheme, S)
    δ  = weno_differences(scheme, S)
    δₛ = weno_differences(scheme, Sₛ)
    ω  = biased_weno_weights(δₛ, scheme, C)
    return weno_reconstruction(scheme, ψ₀, δ, ω, C)
end
