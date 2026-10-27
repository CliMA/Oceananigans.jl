using Oceananigans.Grids: inactive_node, flip_location
using Oceananigans.Utils: newton_div, BackendOptimizedDivision

#####
##### Interpolations to a three-dimensional location along one direction
#####
##### `biased_interpolate_xᶠᶜᶜ(i, j, k, grid, scheme, bias, ψ, args...)` reconstructs `ψ` at `(Face, Center, Center)`
##### from the `(Center, Center, Center)` nodes along `x`. Every scheme reconstructs from the stencil of `load_stencil`.
#####

for (d, ξ) in enumerate((:x, :y, :z)), ℓx in (:ᶜ, :ᶠ), ℓy in (:ᶜ, :ᶠ), ℓz in (:ᶜ, :ᶠ)
    ℓ = map(l -> l == :ᶜ ? Center() : Face(), (ℓx, ℓy, ℓz))
    biased_interpolation    = Symbol(:biased_interpolate_, ξ, ℓx, ℓy, ℓz)
    symmetric_interpolation = Symbol(:symmetric_interpolate_, ξ, ℓx, ℓy, ℓz)

    @eval begin
        @inline $biased_interpolation(i, j, k, grid, scheme, args...)    = biased_interpolate(i, j, k, grid, $(Val(d)), $ℓ, scheme, args...)
        @inline $symmetric_interpolation(i, j, k, grid, scheme, args...) = symmetric_interpolate(i, j, k, grid, $(Val(d)), $ℓ, scheme, args...)
    end
end

#####
##### Stencils continued past walls
#####

@inline stencil_node(i, j, k, ::Val{1}, s) = (i + s, j, k)
@inline stencil_node(i, j, k, ::Val{2}, s) = (i, j + s, k)
@inline stencil_node(i, j, k, ::Val{3}, s) = (i, j, k + s)

@inline node_location(::Val{1}, (ℓx, ℓy, ℓz)) = (flip_location(ℓx), ℓy, ℓz)
@inline node_location(::Val{2}, (ℓx, ℓy, ℓz)) = (ℓx, flip_location(ℓy), ℓz)
@inline node_location(::Val{3}, (ℓx, ℓy, ℓz)) = (ℓx, ℓy, flip_location(ℓz))

@inline stencil_value(i, j, k, grid, ψ, args...)           = @inbounds ψ[i, j, k]
@inline stencil_value(i, j, k, grid, ψ::Callable, args...) = @inbounds ψ(i, j, k, grid, args...)

# φ(a, b) φ(b, c) as the fraction (numerator, denominator), (0, 1) unless a, b and b, c share a sign
@inline function agreement(a, b, c)
    ab, bc = a * b, b * c
    d = @muladd (a * a + b * b) * (b * b + c * c)
    agree = (ab > 0) & (bc > 0) & (d > 0)
    return ifelse(agree, 4ab * bc, zero(a)), ifelse(agree, d, one(a))
end

# the squared agreement factors (n₁ / d₁)² and (n₂ / d₂)²
@inline agreement_squares((n₁, d₁), (n₂, d₂)) = (newton_div(BackendOptimizedDivision, n₁, ifelse(d₁ > floatmin(d₁), d₁, floatmin(d₁)))^2,
                                                 newton_div(BackendOptimizedDivision, n₂, ifelse(d₂ > floatmin(d₂), d₂, floatmin(d₂)))^2)

# one division in Float64, floored at floatmin where d₁ d₂, an eighth power of the differences, would overflow the reciprocal
@inline function agreement_squares((n₁, d₁)::NTuple{2, Float64}, (n₂, d₂)::NTuple{2, Float64})
    D = d₁ * d₂
    r = newton_div(BackendOptimizedDivision, one(D), ifelse(D > floatmin(D), D, floatmin(D)))
    return (n₁ * d₂ * r)^2, (n₂ * d₁ * r)^2
end

# the agreement factors (G₁, G₂) of the nodes `c` inward from a wall, starting with the node next to it, of which `available` have data
@inline function wall_agreement(c, available)
    Δ₁ = ifelse(available ≥ 2, c[2] - c[1], zero(c[1]))
    Δ₂ = ifelse(available ≥ 3, c[3] - c[2], Δ₁)
    Δ₃ = ifelse(available ≥ 4, c[4] - c[3], Δ₂)
    Δ₄ = ifelse(available ≥ 5, c[5] - c[4], Δ₃)
    κ₁, κ₂, κ₃ = Δ₁ - Δ₂, Δ₂ - Δ₃, Δ₃ - Δ₄
    slope_agreement     = agreement(Δ₁, Δ₂, Δ₃)
    curvature_agreement = agreement(κ₁, κ₂, κ₃)
    return agreement_squares(slope_agreement, curvature_agreement)
end

# Second and First order schemes
@inline wall_agreement(c::NTuple{1}, available) = (zero(c[1]), zero(c[1]))

# Third and Fourth order schemes, all the others use the general `wall_agreement`
@inline function wall_agreement(c::NTuple{3}, available)
    Δ₁ = ifelse(available ≥ 2, c[2] - c[1], zero(c[1]))
    Δ₂ = ifelse(available ≥ 3, c[3] - c[2], Δ₁)
    n, d = agreement(Δ₁, Δ₂, Δ₂)
    return newton_div(BackendOptimizedDivision, n, ifelse(d > floatmin(d), d, floatmin(d)))^2, zero(Δ₁)
end

# the node next to the wall, the ghost slope and the ghost curvature, with the agreement factors (G₁, G₂)
@inline function wall_continuation(c, available, (G₁, G₂))
    Δ₁ = ifelse(available ≥ 2, c[2] - c[1], zero(c[1]))
    Δ₂ = ifelse(available ≥ 3, c[3] - c[2], Δ₁)
    σ  = @muladd (G₂ + (1 - G₂) * G₁) * Δ₁
    return c[1], σ, G₂ * (Δ₁ - Δ₂)
end

@inline wall_continuation(c::NTuple{1}, available, G) = (c[1], zero(c[1]), zero(c[1]))

@inline continued_value((c₁, σ, κ), g) = @muladd c₁ - g * (σ + (g + 1) * (κ / 2))

# Tracers continue past walls with the cubic through the nodes next to the wall, c₁ - g δ¹ + g (g + 1) δ² / 2 - g (g + 1) (g + 2) δ³ / 6
# at the g-th node past it, with δ¹, δ², δ³ the forward differences inward, lowered when fewer than four nodes lie between the walls
struct CubicContinuation end

@inline wall_agreements(G::CubicContinuation, args...) = (G, G)

@inline function wall_continuation(c::NTuple{5}, available, ::CubicContinuation)
    δ¹ = ifelse(available ≥ 2, c[2] - c[1], zero(c[1]))
    δ² = ifelse(available ≥ 3, c[3] - 2c[2] + c[1], zero(c[1]))
    δ³ = ifelse(available ≥ 4, c[4] - 3c[3] + 3c[2] - c[1], zero(c[1]))
    return c[1], δ¹, δ², δ³
end

# Third and Fourth order schemes continue the quadratic
@inline function wall_continuation(c::NTuple{3}, available, ::CubicContinuation)
    δ¹ = ifelse(available ≥ 2, c[2] - c[1], zero(c[1]))
    δ² = ifelse(available ≥ 3, c[3] - 2c[2] + c[1], zero(c[1]))
    return c[1], δ¹, δ², zero(c[1])
end

@inline continued_value((c₁, δ¹, δ², δ³)::NTuple{4}, g) = @muladd c₁ - g * δ¹ + g * (g + 1) * δ² / 2 - g * (g + 1) * (g + 2) * δ³ / 6

# The `M` values of `ψ` along `dir` about `(i, j, k)`, the walls `nᴸ` (last node without data in 1:N) and `nᴿ` (first in M-N+1:M), and the nodes `cᴸ` and `cᴿ` inward from each wall
@inline function load_raw_stencil(i, j, k, grid, dir::Val{d}, ℓ, ::Val{N}, ::Val{M}, ψ, args...) where {d, N, M}
    ℓⁿ = node_location(dir, ℓ)
    σ  = ℓ[d] isa Center

    s(n) = n - N - 1 + σ
    # a face node lies past a wall when the cell between it and the point is inactive: a one-cell wall leaves its faces peripheral
    no_data(n) = ifelse(σ, inactive_node(stencil_node(i, j, k, dir, s(n) - (n > N))..., grid, ℓ...),
                           inactive_node(stencil_node(i, j, k, dir, s(n))...,           grid, ℓⁿ...))

    nᴸ = max(Int32(0),     ntuple(n -> ifelse(no_data(n),         Int32(n),         Int32(0)),     Val(N))...)
    nᴿ = min(Int32(M + 1), ntuple(n -> ifelse(no_data(M - N + n), Int32(M - N + n), Int32(M + 1)), Val(N))...)

    S = ntuple(n -> stencil_value(stencil_node(i, j, k, dir, s(n))..., grid, ψ, args...), Val(M))

    # the q-th node inward from each wall, picked along the scan that finds it; no wall shares the values of a wall at 1 or M
    R = Val(min(5, 2N - 1))

    left_wall_positions  = ntuple(n -> n + 1, Val(N - 1))
    right_wall_positions = ntuple(n -> M - n, Val(N - 1))

    left_inward_node(q)  = foldl((c, n) -> ifelse(no_data(n), S[min(n + q, M)], c), left_wall_positions;  init = S[1 + q])
    right_inward_node(q) = foldl((c, n) -> ifelse(no_data(n), S[max(n - q, 1)], c), right_wall_positions; init = S[M - q])

    cᴸ = ntuple(left_inward_node,  R)
    cᴿ = ntuple(right_inward_node, R)

    return S, nᴸ, nᴿ, cᴸ, cᴿ
end

@inline function wall_agreements(::Nothing, cᴸ, cᴿ, available)
    Gᴸ = wall_agreement(cᴸ, available)
    Gᴿ = wall_agreement(cᴿ, available)
    return Gᴸ, Gᴿ
end

@inline wall_agreements(G::Tuple, args...) = G

# The `2N` values `S` with the nodes past the walls `nᴸ` and `nᴿ` replaced by ghosts continued from the nodes `cᴸ` and `cᴿ`, with
# the agreement factors `G = (Gᴸ, Gᴿ)`, their own when `nothing`; returns the stencil and the factors
@inline function fill_stencil_ghosts(S, nᴸ, nᴿ, cᴸ, cᴿ, G, ::Val{N}) where N
    available = nᴿ - Int32(1) - nᴸ

    Gᴸ, Gᴿ = wall_agreements(G, cᴸ, cᴿ, available)
    left   = wall_continuation(cᴸ, available, Gᴸ)
    right  = wall_continuation(cᴿ, available, Gᴿ)

    # the ghost index as a float, converted once per wall
    Lᴸ = convert(eltype(S), nᴸ + Int32(1))
    Lᴿ = convert(eltype(S), nᴿ - Int32(1))
    left_ghost(n)  = continued_value(left,  Lᴸ - n)
    right_ghost(n) = continued_value(right, n - Lᴿ)

    stencil = ntuple(n -> ifelse(n ≤ nᴸ, left_ghost(n), ifelse(n ≥ nᴿ, right_ghost(n), S[n])), Val(2N))

    return stencil, (Gᴸ, Gᴿ)
end

# `ψ` continued past walls with the agreement factors `G = (Gᴸ, Gᴿ)`, its own when `nothing`; returns the stencil and the factors
@inline function load_stencil(i, j, k, grid, dir, ℓ, ::Val{N}, G::Union{Nothing, Tuple}, ψ, args...) where N
    S, nᴸ, nᴿ, cᴸ, cᴿ = load_raw_stencil(i, j, k, grid, dir, ℓ, Val(N), Val(2N), ψ, args...)
    return fill_stencil_ghosts(S, nᴸ, nᴿ, cᴸ, cᴿ, G, Val(N))
end

"""
$(TYPEDSIGNATURES)

Return the `2N` values of `ψ` that reconstruct `ψ` at `(i, j, k)`, located at `ℓ`, along the direction `dir`.

Scanning outward from `(i, j, k)` on each side, the first node without data marks a wall. Past the wall the g-th node takes
the ghost

```math
c₁ - g σ - g (g + 1) κ / 2, \\quad σ = [G₂ + (1 - G₂) G₁] Δ₁, \\quad κ = G₂ κ₁,
\\quad G₁ = [φ(Δ₁, Δ₂) φ(Δ₂, Δ₃)]², \\quad G₂ = [φ(κ₁, κ₂) φ(κ₂, κ₃)]²,
```

where `c₁` is the node with data next to the wall, `Δⱼ` the differences inward from it, `κⱼ = Δⱼ - Δⱼ₊₁`, and
`φ(a, b) = 2ab / (a² + b²)` when `a` and `b` share a sign, zero otherwise. Linear data, and quadratic data on five nodes, are
continued exactly, and next to a front (`G₁ = G₂ = 0`) the ghosts are constant.
"""
@inline function load_stencil(i, j, k, grid, dir, ℓ, ::Val{N}, ψ, args...) where N
    S, nᴸ, nᴿ, cᴸ, cᴿ = load_raw_stencil(i, j, k, grid, dir, ℓ, Val(N), Val(2N), ψ, args...)
    S, _ = fill_stencil_ghosts(S, nᴸ, nᴿ, cᴸ, cᴿ, nothing, Val(N))
    return S
end

# a stencil with fewer than five nodes past a wall has no curvature factor
@inline face_agreement((G₁, G₂), available) = (G₁, ifelse(available ≥ 5, G₂, zero(G₂)))
@inline face_agreement(G::CubicContinuation, available) = G

# The stencils of the point `(i, j, k)` and of the next point along `dir` with their walls, from one window of their `M = 2N + 1`
# values, continued with one agreement per wall or with `G = CubicContinuation()`
@inline load_stencil(i, j, k, grid, dir, ℓ, ::Val{N}, ::Val{M}, ψ, args...) where {N, M} = load_stencil(i, j, k, grid, dir, ℓ, Val(N), Val(M), nothing, ψ, args...)

@inline function load_stencil(i, j, k, grid, dir, ℓ, ::Val{N}, ::Val{M}, G::Union{Nothing, CubicContinuation}, ψ, args...) where {N, M}
    W, nᴸ, nᴿ, cᴸ, cᴿ = load_raw_stencil(i, j, k, grid, dir, ℓ, Val(N), Val(M), ψ, args...)
    Gᴸ, Gᴿ = wall_agreements(G, cᴸ, cᴿ, nᴿ - Int32(1) - nᴸ)

    # the point (i, j, k) reconstructs from the first 2N values, the next point from the last 2N
    W⁻ = ntuple(n -> W[n],     Val(2N))
    W⁺ = ntuple(n -> W[n + 1], Val(2N))
    nᴿ⁻ = min(nᴿ, Int32(M))
    nᴸ⁺, nᴿ⁺ = max(nᴸ - Int32(1), Int32(0)), nᴿ - Int32(1)

    G⁻ = (face_agreement(Gᴸ, nᴿ⁻ - Int32(1) - nᴸ), Gᴿ)
    G⁺ = (Gᴸ, face_agreement(Gᴿ, nᴿ⁺ - Int32(1) - nᴸ⁺))

    S⁻, _ = fill_stencil_ghosts(W⁻, nᴸ, nᴿ⁻, cᴸ, cᴿ, G⁻, Val(N))
    S⁺, _ = fill_stencil_ghosts(W⁺, nᴸ⁺, nᴿ⁺, cᴸ, cᴿ, G⁺, Val(N))

    return (S⁻, nᴸ, nᴿ⁻), (S⁺, nᴸ⁺, nᴿ⁺)
end

#####
##### Tracers next to walls
#####

@inline linear_interpolate(scheme, bias, S) = biased_interpolate(scheme, bias, S)

"""
$(TYPEDSIGNATURES)

Return the tracer value at a face from its stencil `S`, continued past the walls `nᴸ` and `nᴿ` with the cubic continuation of
`load_stencil`. The first face from a wall takes the centered value of its two nodes. A face whose upwind stencil (its
`2N - 1` upwind nodes for `bias`, all `2N` nodes for centered schemes) reaches past a wall takes the linear reconstruction of
`scheme`, which for `WENO` is the upwind-biased reconstruction of the same order. Every other face takes the reconstruction of
`scheme`.
"""
@inline function wall_interpolate(scheme::AbstractAdvectionScheme{N}, bias, (S, nᴸ, nᴿ)) where N
    first_face = (N > 1) & ((nᴸ == Int32(N - 1)) | (nᴿ == Int32(N + 2)))
    past_left  = (nᴸ ≥ Int32(1)) | (nᴿ ≤ Int32(2N - 1))
    past_right = (nᴸ ≥ Int32(2)) | (nᴿ ≤ Int32(2N))
    past_wall  = ifelse(bias == LeftBias, past_left, ifelse(bias == RightBias, past_right, past_left | past_right))
    ψˡ = linear_interpolate(scheme, bias, S)
    ψ̂  = biased_interpolate(scheme, bias, S)
    return ifelse(first_face, (S[N] + S[N + 1]) / 2, ifelse(past_wall, ψˡ, ψ̂))
end

@inline function tracer_interpolate(i, j, k, grid, dir, ℓ, scheme::AbstractAdvectionScheme{N}, bias, c) where N
    S, nᴸ, nᴿ, cᴸ, cᴿ = load_raw_stencil(i, j, k, grid, dir, ℓ, Val(N), Val(2N), c)
    S, _ = fill_stencil_ghosts(S, nᴸ, nᴿ, cᴸ, cᴿ, CubicContinuation(), Val(N))
    return wall_interpolate(scheme, bias, (S, nᴸ, nᴿ))
end
